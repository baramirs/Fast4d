"""BVM indexing motor: RANSAC lattice + hkl assignment with zone-axis anchoring.

Pure numpy (+ py4DSTEM get_maxima_2D). No Qt. Ported from
``notebooks/indexing_bvm_demo.ipynb`` (sections 5–9) and GPA
``gpa/indexing/lattice.py`` (score/refine/reduce/RANSAC), without importing GPA.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PeakRow:
    """One BVM maximum with indexing columns."""
    qx: float
    qy: float
    qx_abs_px: float
    qy_abs_px: float
    gx: float
    gy: float
    m: int
    n: int
    h: int
    k: int
    l: int
    d_exp: float
    d_theo: float
    dd_pct: float
    intensity: float
    residual_px: float
    ok: bool
    peak_index: int = -1  # index in get_maxima_2D list


@dataclass
class IndexingResult:
    """Full BVM indexing result (table + bases + metrics)."""
    peaks: list[PeakRow]
    origin_px: np.ndarray          # (qx, qy) absolute BVM px
    basis_a_A: np.ndarray          # primitive RANSAC vector a (Å⁻¹)
    basis_b_A: np.ndarray
    basis_a_px: np.ndarray
    basis_b_px: np.ndarray
    g1_hkl: np.ndarray             # anchored Miller indices for g1 pick
    g2_hkl: np.ndarray
    a_hkl: np.ndarray              # primitive hkl
    b_hkl: np.ndarray
    zone_axis: np.ndarray
    real_axis_h: np.ndarray
    real_axis_v: np.ndarray
    Q_pixel: float
    Q_units: str
    tol_px: float
    index_origin: int
    index_g1: int
    index_g2: int
    g1_px: np.ndarray              # proposed g1 relative to origin (px)
    g2_px: np.ndarray
    n_inliers: int = 0
    angle_deg: float = 0.0
    lattice_a: float = 0.0
    qr_sign: int = 1
    metrics: dict[str, Any] = field(default_factory=dict)
    bvm: np.ndarray | None = None  # optional for overlay (not serialized)

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "peak_index", "qx", "qy", "qx_abs_px", "qy_abs_px",
            "gx", "gy", "m", "n", "h", "k", "l",
            "d_exp", "d_theo", "dd_pct", "intensity", "residual_px", "ok",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for p in self.peaks:
                w.writerow({k: getattr(p, k) for k in fields})
        return path


# ── Maxima (same contract as StrainMap.choose_basis_vectors) ──────────────────

def find_bvm_maxima(
    bvm: np.ndarray,
    *,
    min_spacing: float = 10,
    min_absolute_intensity: float = 80,
    max_num_peaks: int = 60,
    edge_boundary: float = 4,
    subpixel: str = "multicorr",
    upsample_factor: int = 16,
    sigma: float = 0,
    image_upsample: int = 1,
) -> np.ndarray:
    """Detect maxima on a BVM (delegates to ``plugins.indexing.peaks.find_peaks``).

    ``image_upsample`` (1|2|4) zooms the BVM before detection, then scales
    coordinates back — distinct from multicorr ``upsample_factor``.

    Returns a structured array with fields ``x``, ``y``, ``intensity``, sorted by
    intensity descending — the index order that ``index_origin/g1/g2`` refer to.
    """
    from plugins.indexing.peaks import find_peaks

    return find_peaks(
        bvm,
        min_spacing=min_spacing,
        min_absolute_intensity=min_absolute_intensity,
        max_num_peaks=max_num_peaks,
        edge_boundary=edge_boundary,
        subpixel=subpixel,
        upsample_factor=upsample_factor,
        sigma=sigma,
        image_upsample=image_upsample,
    )


# ── RANSAC lattice (ported from GPA gpa/indexing/lattice.py) ──────────────────

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def score_basis(
    points: np.ndarray,
    weights: np.ndarray,
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    tol: float,
    max_index_2d: int = 5,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Assign integer (m, n) and score inliers. Returns score, mask, residuals."""
    basis = np.column_stack([vec_a, vec_b])
    try:
        coeffs = np.linalg.solve(basis, points.T).T
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros(len(points), dtype=bool), np.full(len(points), np.inf)

    rounded = np.round(coeffs)
    valid = (
        (np.abs(rounded[:, 0]) <= max_index_2d)
        & (np.abs(rounded[:, 1]) <= max_index_2d)
        & ~((rounded[:, 0] == 0) & (rounded[:, 1] == 0))
    )
    pred = rounded @ basis.T
    residuals = np.linalg.norm(points - pred, axis=1)
    inliers = valid & (residuals <= tol)
    if not np.any(inliers):
        return -np.inf, inliers, residuals
    score = float(np.sum(weights[inliers] * (1.0 - residuals[inliers] / (tol + 1e-15))))
    return score, inliers, residuals


def refine_basis(
    points: np.ndarray,
    weights: np.ndarray,
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    tol: float,
    max_index_2d: int = 5,
    n_iter: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Iterative weighted least-squares refinement of the reciprocal basis."""
    a, b = vec_a.copy(), vec_b.copy()
    residuals = np.full(len(points), np.inf)
    inliers = np.zeros(len(points), dtype=bool)
    score = -np.inf
    for _ in range(n_iter):
        score, inliers, residuals = score_basis(
            points, weights, a, b, tol=tol, max_index_2d=max_index_2d
        )
        if int(np.sum(inliers)) < 3:
            break
        coeffs = np.round(np.linalg.solve(np.column_stack([a, b]), points[inliers].T).T)
        try:
            bt, *_ = np.linalg.lstsq(coeffs, points[inliers], rcond=None)
            a, b = bt[0], bt[1]
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(a) < 1e-15 or np.linalg.norm(b) < 1e-15:
            break
    score, inliers, residuals = score_basis(
        points, weights, a, b, tol=tol, max_index_2d=max_index_2d
    )
    return a, b, inliers, residuals, score


def reduce_basis_2d(
    vec_a: np.ndarray, vec_b: np.ndarray, n_iter: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss lattice reduction in 2D."""
    a = np.asarray(vec_a, dtype=np.float64).copy()
    b = np.asarray(vec_b, dtype=np.float64).copy()
    for _ in range(n_iter):
        if np.linalg.norm(a) > np.linalg.norm(b):
            a, b = b, a
        na2 = float(np.dot(a, a))
        if na2 < 1e-18:
            break
        mu = round(float(np.dot(b, a) / na2))
        b = b - mu * a
        if np.linalg.norm(b) >= np.linalg.norm(a) - 1e-12 and mu == 0:
            break
    if np.linalg.norm(b) < 1e-15:
        return a, np.asarray(vec_b, dtype=np.float64)
    if np.linalg.norm(a) < np.linalg.norm(b):
        return a, b
    return b, a


def fit_lattice_ransac(
    points_A: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    tol_A: float | None = None,
    n_candidates: int | None = None,
    n_iterations: int = 3000,
    max_index_2d: int = 5,
    min_angle_deg: float = 20.0,
    max_angle_deg: float = 160.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a 2D reciprocal lattice via pair-sampling RANSAC.

    ``points_A`` is (N, 2) in physical reciprocal units (Å⁻¹). Returns dict with
    ``vector_a``, ``vector_b``, ``inliers``, ``residuals``, ``score``, ``angle_deg``.
    """
    points = np.asarray(points_A, dtype=np.float64)
    n = len(points)
    if n < 4:
        raise ValueError("Need at least 4 points for lattice fitting")

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    # Rank by weight and keep top N
    order = np.argsort(weights)[::-1]
    if n_candidates is None:
        n_candidates = n
    keep = order[: min(int(n_candidates), n)]
    points = points[keep]
    weights = weights[keep].copy()
    weights /= max(float(weights.max()), 1e-15)
    n = len(points)
    index_map = keep  # original indices of kept points

    if tol_A is None:
        mags = np.linalg.norm(points, axis=1)
        tol_A = 0.05 * float(np.median(mags[mags > 0])) if np.any(mags > 0) else 0.02
    tol = float(tol_A)

    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best = None

    for _ in range(n_iterations):
        i, j = rng.choice(n, size=2, replace=False)
        va, vb = points[i].copy(), points[j].copy()
        ang = _angle_deg(va, vb)
        if ang < min_angle_deg or ang > max_angle_deg:
            continue
        if np.linalg.norm(va) < 1e-9 or np.linalg.norm(vb) < 1e-9:
            continue
        for sa in (1.0, -1.0):
            for sb in (1.0, -1.0):
                a2, b2, inliers, residuals, _score = refine_basis(
                    points, weights, sa * va, sb * vb, tol=tol, max_index_2d=max_index_2d
                )
                a2r, b2r = reduce_basis_2d(a2, b2)
                score_r, inliers_r, residuals_r = score_basis(
                    points, weights, a2r, b2r, tol=tol, max_index_2d=max_index_2d
                )
                ang_r = _angle_deg(a2r, b2r)
                bonus = 0.05 * score_r * (1.0 - abs(ang_r - 90.0) / 90.0)
                score_r = score_r + max(bonus, 0.0)
                if score_r > best_score:
                    best_score = score_r
                    best = (a2r, b2r, inliers_r, residuals_r, score_r)

    if best is None:
        raise RuntimeError("RANSAC failed to find a valid reciprocal basis")

    a, b, inliers, residuals, score = best
    a, b = reduce_basis_2d(a, b)
    score, inliers, residuals = score_basis(
        points, weights, a, b, tol=tol, max_index_2d=max_index_2d
    )
    if np.linalg.norm(b) > np.linalg.norm(a):
        a, b = b, a
    if np.dot(a, b) < 0:
        b = -b

    return {
        "vector_a": np.asarray(a, dtype=np.float64),
        "vector_b": np.asarray(b, dtype=np.float64),
        "inliers": np.asarray(inliers, dtype=bool),
        "residuals": np.asarray(residuals, dtype=np.float64),
        "score": float(score),
        "angle_deg": _angle_deg(a, b),
        "tol": tol,
        "kept_indices": np.asarray(index_map, dtype=int),
    }


# ── hkl assignment + absolute anchoring ───────────────────────────────────────

def _hkl_str(v: Sequence[int | float]) -> str:
    return "(" + " ".join(str(int(x)) for x in v) + ")"


def enumerate_zolz(zone_axis: np.ndarray, max_index: int = 4) -> np.ndarray:
    """ZOLZ reflections satisfying Weiss zone law up to ``max_index``."""
    za = np.asarray(zone_axis, dtype=int).ravel()[:3]
    rng = range(-max_index, max_index + 1)
    rows = [
        (h, k, l)
        for h in rng for k in rng for l in rng
        if (h, k, l) != (0, 0, 0) and h * za[0] + k * za[1] + l * za[2] == 0
    ]
    return np.asarray(rows, dtype=int)


def match_g1g2_hkl(
    g1_A: np.ndarray,
    g2_A: np.ndarray,
    zone_axis: np.ndarray,
    lattice_a: float,
    *,
    max_index: int = 4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Preliminary (G1, G2) by |g|/angle/chirality match against ZOLZ."""
    zone_axis = np.asarray(zone_axis, dtype=float).ravel()[:3]
    zolz = enumerate_zolz(zone_axis, max_index=max_index)
    g_theo = np.linalg.norm(zolz.astype(float), axis=1) / float(lattice_a)
    mag1 = float(np.hypot(*g1_A))
    mag2 = float(np.hypot(*g2_A))
    meas_ang = _angle_deg(g1_A, g2_A)
    # chirality in detector space from px vectors passed as g*_A here is fine
    # when caller passes consistent orientation; we use 2D cross of Å⁻¹ vectors
    meas_chir = float(np.sign(g1_A[0] * g2_A[1] - g1_A[1] * g2_A[0]))

    best = None
    for i1 in np.argsort(np.abs(g_theo - mag1))[:8]:
        for i2 in np.argsort(np.abs(g_theo - mag2))[:8]:
            G1c = zolz[i1].astype(float)
            G2c = zolz[i2].astype(float)
            cr = np.cross(G1c, G2c)
            if np.linalg.norm(cr) < 1e-9:
                continue
            cost = (
                abs(g_theo[i1] - mag1) / max(mag1, 1e-15)
                + abs(g_theo[i2] - mag2) / max(mag2, 1e-15)
                + abs(_angle_deg(G1c, G2c) - meas_ang) / 90.0
            )
            if np.sign(np.dot(cr, zone_axis)) != meas_chir:
                cost += 0.5
            if best is None or cost < best[0]:
                best = (cost, zolz[i1].copy(), zolz[i2].copy())
    if best is None:
        raise RuntimeError("No ZOLZ (G1, G2) pair matched measured basis")
    return best[1], best[2], float(best[0])


def anchor_hkl_with_real_axes(
    g1_px: np.ndarray,
    g2_px: np.ndarray,
    mag1_A: float,
    mag2_A: float,
    *,
    zone_axis: np.ndarray,
    real_axis_h: np.ndarray,
    real_axis_v: np.ndarray,
    qr_rotation_deg: float,
    qr_flip: bool = False,
    lattice_a: float,
    prelim_g1_hkl: np.ndarray | None = None,
    prelim_g2_hkl: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Absolute sign anchoring via user real-space axes + QR rotation.

    Returns ``(G1_hkl, G2_hkl, qr_sign)`` where ``qr_sign`` is ±1 for the
    empirical QR convention that aligns both axes.
    """
    zone_axis = np.asarray(zone_axis, dtype=float).ravel()[:3]
    real_axis_h = np.asarray(real_axis_h, dtype=float).ravel()[:3]
    real_axis_v = np.asarray(real_axis_v, dtype=float).ravel()[:3]
    g1_px = np.asarray(g1_px, dtype=float).ravel()[:2]
    g2_px = np.asarray(g2_px, dtype=float).ravel()[:2]

    real_axes = [
        (real_axis_h, np.array([0.0, 1.0]), "horizontal (+ry)"),
        (real_axis_v, np.array([1.0, 0.0]), "vertical (+rx)"),
    ]
    meas_basis = [
        (g1_px, mag1_A, "g1"),
        (-g1_px, mag1_A, "-g1"),
        (g2_px, mag2_A, "g2"),
        (-g2_px, mag2_A, "-g2"),
    ]

    def _match_axis(q_hat, dir3):
        cands = []
        for vec, mag, vname in meas_basis:
            n = mag * lattice_a / max(float(np.linalg.norm(dir3)), 1e-15)
            if abs(n - round(n)) < 0.1 and 1 <= round(n) <= 4:
                cands.append((_angle_deg(q_hat, vec), vname, vec, int(round(n))))
        cands.sort(key=lambda t: t[0])
        return cands[0] if cands else None

    solution = None
    for s in (+1, -1):
        th = np.radians(s * float(qr_rotation_deg))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        matches, ok = [], True
        for dir3, r_hat, _label in real_axes:
            q_hat = R @ r_hat
            # py4DSTEM QR_flip: swap Q components after rotation (see StrainMap axes)
            if qr_flip:
                q_hat = np.array([q_hat[1], q_hat[0]], dtype=float)
            m = _match_axis(q_hat, dir3)
            if m is None or m[0] > 5.0:
                ok = False
                break
            matches.append((dir3, m))
        if ok:
            solution = (s, matches)
            break
    if solution is None:
        raise RuntimeError(
            "No QR rotation sign aligns both real axes — check zone/real axes "
            f"(zone={_hkl_str(zone_axis)}, "
            f"real_H={_hkl_str(real_axis_h)}, real_V={_hkl_str(real_axis_v)}, "
            f"QR={float(qr_rotation_deg):.3f}°, QR_flip={bool(qr_flip)}). "
            "If orientation is unknown, run Index BVM in Unknown orientation mode "
            "(lattice + g1/g2 proposal without absolute hkl)."
        )

    s, matches = solution
    anchored: dict[str, np.ndarray] = {}
    for dir3, (ang_err, vname, _vec, n) in matches:
        key = vname.lstrip("-")
        G_spot = n * dir3 if not vname.startswith("-") else -n * dir3
        anchored[key] = np.asarray(G_spot, dtype=int)
        _ = ang_err  # used for diagnostics by callers via metrics

    if set(anchored) != {"g1", "g2"}:
        raise RuntimeError("Both real axes must map to g1 and g2")

    G1 = anchored["g1"].astype(int)
    G2 = anchored["g2"].astype(int)
    if prelim_g1_hkl is not None and prelim_g2_hkl is not None:
        # Prefer physical anchoring when it differs from magnitude match
        pass
    if int(np.dot(G1, zone_axis)) != 0 or int(np.dot(G2, zone_axis)) != 0:
        raise RuntimeError(
            f"Anchored G1/G2 violate zone law: G1={_hkl_str(G1)} G2={_hkl_str(G2)}"
        )
    return G1, G2, int(s)


def propose_basis_indices(
    qx: np.ndarray,
    qy: np.ndarray,
    g1_target_px: np.ndarray,
    g2_target_px: np.ndarray,
) -> tuple[int, int, int]:
    """Map origin + two target vectors to indices in the maxima list.

    Origin = closest to (0,0); g1/g2 = closest to the target relative vectors.
    """
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    i0 = int(np.argmin(np.hypot(qx, qy)))
    i1 = int(np.argmin(np.hypot(qx - g1_target_px[0], qy - g1_target_px[1])))
    i2 = int(np.argmin(np.hypot(qx - g2_target_px[0], qy - g2_target_px[1])))
    return i0, i1, i2


def select_orthogonal_g_pair(
    a_px: np.ndarray,
    b_px: np.ndarray,
    *,
    target_angle_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick g1,g2 as short combinations of primitive (a,b) near ``target_angle``.

    Candidates: ±a, ±b, ±(a+b), ±(a−b). Returns ``(g1, g2, M)`` where columns of
    ``M`` express g1,g2 in the primitive basis (integer, det ≠ 0).
    """
    a = np.asarray(a_px, dtype=float).ravel()[:2]
    b = np.asarray(b_px, dtype=float).ravel()[:2]
    cands = []
    for m, n in ((1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1), (-1, 0), (0, -1)):
        v = m * a + n * b
        if np.linalg.norm(v) < 1e-9:
            continue
        cands.append((np.array([m, n], dtype=int), v))

    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            ci, vi = cands[i]
            cj, vj = cands[j]
            M = np.column_stack([ci, cj])
            if abs(int(round(np.linalg.det(M)))) == 0:
                continue
            ang = _angle_deg(vi, vj)
            if ang < 25.0 or ang > 155.0:
                continue
            ni, nj = float(np.linalg.norm(vi)), float(np.linalg.norm(vj))
            aniso = abs(ni - nj) / max(0.5 * (ni + nj), 1e-15)
            cost = abs(ang - target_angle_deg) / 90.0 + 0.35 * aniso
            # Prefer longer (stronger) reflections slightly for strain picks
            cost -= 0.02 * (ni + nj) / max(ni + nj, 1e-9)
            if best is None or cost < best[0]:
                best = (cost, vi, vj, M)
    if best is None:
        # fallback: a and b themselves
        M = np.eye(2, dtype=int)
        return a.copy(), b.copy(), M
    return best[1].copy(), best[2].copy(), best[3]


# ── Top-level indexing ────────────────────────────────────────────────────────

ORIENTATION_MODES = ("unknown", "known")


def _normalize_orientation_mode(mode: str | None) -> str:
    m = str(mode or "known").strip().lower()
    if m not in ORIENTATION_MODES:
        raise ValueError(f"orientation_mode must be one of {ORIENTATION_MODES}, got {mode!r}")
    return m


def index_bvm(
    bvm: np.ndarray,
    origin_px: np.ndarray,
    *,
    Q_pixel: float,
    Q_units: str = "A^-1",
    lattice_a: float = 5.4309,
    zone_axis: Sequence[int] | None = (1, 1, 0),
    real_axis_h: Sequence[int] = (0, 0, -1),
    real_axis_v: Sequence[int] = (-1, 1, 0),
    qr_rotation_deg: float = 135.0,
    qr_flip: bool = False,
    tol_px: float = 2.0,
    seed: int = 0,
    min_spacing: float = 20,
    min_absolute_intensity: float = 80,
    max_num_peaks: int = 60,
    edge_boundary: float = 40,
    subpixel: str = "multicorr",
    maxima: np.ndarray | None = None,
    orientation_mode: str = "known",
    image_upsample: int = 1,
) -> IndexingResult:
    """Run BVM indexing: maxima → RANSAC → (optional) hkl → propose Basis indices.

    ``orientation_mode``:
      * ``\"known\"`` — absolute hkl via zone + real axes + QR (notebook path).
      * ``\"unknown\"`` — lattice + g1/g2 proposal only; optional relative hkl if
        ``zone_axis`` is provided (signs not anchored). Never invents orientation.

    ``origin_px`` is the calibrated origin in absolute BVM pixels (qx, qy).
    Physical units come from ``Q_pixel`` (Å⁻¹/px from braggpeaks.calibration).
    """
    mode = _normalize_orientation_mode(orientation_mode)
    bvm = np.asarray(bvm, dtype=float)
    origin_px = np.asarray(origin_px, dtype=float).ravel()[:2]
    has_zone = zone_axis is not None and len(list(zone_axis)) >= 3
    zone = np.asarray(zone_axis if has_zone else (0, 0, 0), dtype=int).ravel()[:3]
    rax_h = np.asarray(real_axis_h, dtype=int).ravel()[:3]
    rax_v = np.asarray(real_axis_v, dtype=int).ravel()[:3]
    Q = float(Q_pixel)
    if Q <= 0:
        raise ValueError("Q_pixel must be > 0")
    if mode == "known" and not has_zone:
        raise ValueError("Known orientation mode requires zone_axis [uvw]")

    if maxima is None:
        maxima = find_bvm_maxima(
            bvm,
            min_spacing=min_spacing,
            min_absolute_intensity=min_absolute_intensity,
            max_num_peaks=max_num_peaks,
            edge_boundary=edge_boundary,
            subpixel=subpixel,
            image_upsample=int(image_upsample),
        )
    gx_abs = np.asarray(maxima["x"], dtype=float)
    gy_abs = np.asarray(maxima["y"], dtype=float)
    intensity = np.asarray(maxima["intensity"], dtype=float)
    qx = gx_abs - origin_px[0]
    qy = gy_abs - origin_px[1]
    n_peaks = len(qx)

    gx_A = qx * Q
    gy_A = qy * Q
    points_A = np.column_stack([gx_A, gy_A])
    tol_A = float(tol_px) * Q

    # Exclude near-origin from RANSAC sampling weights slightly? Keep all; score_basis
    # already rejects (0,0). Use intensity as weight.
    lat = fit_lattice_ransac(
        points_A,
        weights=intensity,
        tol_A=tol_A,
        n_candidates=n_peaks,
        seed=int(seed),
    )
    a_A = lat["vector_a"]
    b_A = lat["vector_b"]
    a_px = a_A / Q
    b_px = b_A / Q

    # Index all peaks in primitive basis (full list, not only kept)
    B_px = np.column_stack([a_px, b_px])
    try:
        coeff = np.linalg.solve(B_px, np.column_stack([qx, qy]).T).T
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"Singular primitive basis: {exc}") from exc
    mn = np.round(coeff).astype(int)
    pred = mn @ B_px.T
    res_px = np.hypot(qx - pred[:, 0], qy - pred[:, 1])
    ok = (res_px < float(tol_px)) & ~((mn[:, 0] == 0) & (mn[:, 1] == 0))
    # Origin peak: treat residual against (0,0)
    i0_guess = int(np.argmin(np.hypot(qx, qy)))
    ok[i0_guess] = res_px[i0_guess] < float(tol_px)  # allow origin as ok separately

    # Choose orthogonal-ish g1/g2 for strain (like manifest pick), then optional hkl
    g1_cand, g2_cand, M = select_orthogonal_g_pair(a_px, b_px, target_angle_deg=90.0)
    g1_A = g1_cand * Q
    g2_A = g2_cand * Q
    mag1 = float(np.hypot(*g1_A))
    mag2 = float(np.hypot(*g2_A))

    match_cost = float("nan")
    anchored = False
    qr_sign = 0
    G1 = np.zeros(3, dtype=int)
    G2 = np.zeros(3, dtype=int)
    A_hkl = np.zeros(3, dtype=int)
    B_hkl = np.zeros(3, dtype=int)
    assign_hkl = False

    if mode == "known":
        G1_pre, G2_pre, match_cost = match_g1g2_hkl(
            g1_A, g2_A, zone, float(lattice_a)
        )
        G1, G2, qr_sign = anchor_hkl_with_real_axes(
            g1_cand, g2_cand, mag1, mag2,
            zone_axis=zone,
            real_axis_h=rax_h,
            real_axis_v=rax_v,
            qr_rotation_deg=float(qr_rotation_deg),
            qr_flip=bool(qr_flip),
            lattice_a=float(lattice_a),
            prelim_g1_hkl=G1_pre,
            prelim_g2_hkl=G2_pre,
        )
        anchored = True
        assign_hkl = True
    elif has_zone:
        # Unknown: optional relative Miller labels (sign-ambiguous), no absolute anchor
        try:
            G1, G2, match_cost = match_g1g2_hkl(
                g1_A, g2_A, zone, float(lattice_a)
            )
            assign_hkl = True
        except Exception:
            G1 = np.zeros(3, dtype=int)
            G2 = np.zeros(3, dtype=int)
            match_cost = float("nan")
            assign_hkl = False

    if assign_hkl:
        # Primitive hkl from G1/G2 and M: [A B] = [G1 G2] · M^{-1}
        Minv = np.linalg.inv(M.astype(float))
        AB = np.column_stack([G1, G2]).astype(float) @ Minv
        if not np.allclose(AB, np.round(AB), atol=0.15):
            A_pre, B_pre, _ = match_g1g2_hkl(a_A, b_A, zone, float(lattice_a))
            A_hkl = np.round(AB[:, 0]).astype(int)
            B_hkl = np.round(AB[:, 1]).astype(int)
            if np.linalg.norm(A_hkl) < 1e-9 or np.linalg.norm(B_hkl) < 1e-9:
                A_hkl, B_hkl = A_pre.astype(int), B_pre.astype(int)
        else:
            A_hkl = np.round(AB[:, 0]).astype(int)
            B_hkl = np.round(AB[:, 1]).astype(int)
        hkl_all = mn @ np.stack([A_hkl, B_hkl])
    else:
        hkl_all = np.zeros((n_peaks, 3), dtype=int)

    # Origin → (0,0,0)
    hkl_all[i0_guess] = (0, 0, 0)

    g_mag_A = np.hypot(gx_A, gy_A)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_exp = np.where(g_mag_A > 0, 1.0 / g_mag_A, np.inf)
        if assign_hkl:
            norm_hkl = np.linalg.norm(hkl_all.astype(float), axis=1)
            d_theo = np.where(norm_hkl > 0, float(lattice_a) / norm_hkl, np.inf)
            dd_pct = np.where(
                np.isfinite(d_theo) & (d_theo > 0),
                100.0 * (d_exp - d_theo) / d_theo,
                np.nan,
            )
        else:
            d_theo = np.full(n_peaks, np.inf)
            dd_pct = np.full(n_peaks, np.nan)

    # Origin always ok if closest
    ok_final = ok.copy()
    ok_final[i0_guess] = True
    if assign_hkl and has_zone:
        zone_law = hkl_all @ zone.astype(int)
        bad_zone = ok_final & (zone_law != 0) & (np.arange(n_peaks) != i0_guess)
        ok_final[bad_zone] = False

    index_origin, index_g1, index_g2 = propose_basis_indices(qx, qy, g1_cand, g2_cand)
    # Use actual peak positions as proposed g1/g2 (relative)
    g1_px = np.array([qx[index_g1], qy[index_g1]], dtype=float)
    g2_px = np.array([qx[index_g2], qy[index_g2]], dtype=float)

    peaks: list[PeakRow] = []
    for i in range(n_peaks):
        peaks.append(PeakRow(
            qx=float(qx[i]), qy=float(qy[i]),
            qx_abs_px=float(gx_abs[i]), qy_abs_px=float(gy_abs[i]),
            gx=float(gx_A[i]), gy=float(gy_A[i]),
            m=int(mn[i, 0]), n=int(mn[i, 1]),
            h=int(hkl_all[i, 0]), k=int(hkl_all[i, 1]), l=int(hkl_all[i, 2]),
            d_exp=float(d_exp[i]) if np.isfinite(d_exp[i]) else float("inf"),
            d_theo=float(d_theo[i]) if np.isfinite(d_theo[i]) else float("inf"),
            dd_pct=float(dd_pct[i]) if np.isfinite(dd_pct[i]) else float("nan"),
            intensity=float(intensity[i]),
            residual_px=float(res_px[i]),
            ok=bool(ok_final[i]),
            peak_index=int(i),
        ))

    return IndexingResult(
        peaks=peaks,
        origin_px=origin_px.copy(),
        basis_a_A=a_A.copy(),
        basis_b_A=b_A.copy(),
        basis_a_px=a_px.copy(),
        basis_b_px=b_px.copy(),
        g1_hkl=G1.copy(),
        g2_hkl=G2.copy(),
        a_hkl=A_hkl.copy(),
        b_hkl=B_hkl.copy(),
        zone_axis=zone.copy(),
        real_axis_h=rax_h.copy(),
        real_axis_v=rax_v.copy(),
        Q_pixel=Q,
        Q_units=str(Q_units),
        tol_px=float(tol_px),
        index_origin=int(index_origin),
        index_g1=int(index_g1),
        index_g2=int(index_g2),
        g1_px=g1_px,
        g2_px=g2_px,
        n_inliers=int(ok_final.sum()),
        angle_deg=float(lat["angle_deg"]),
        lattice_a=float(lattice_a),
        qr_sign=int(qr_sign),
        metrics={
            "match_cost": float(match_cost) if match_cost == match_cost else float("nan"),
            "ransac_score": float(lat["score"]),
            "M": M.tolist(),
            "n_peaks": int(n_peaks),
            "g1_hkl_str": _hkl_str(G1) if assign_hkl else "—",
            "g2_hkl_str": _hkl_str(G2) if assign_hkl else "—",
            "a_hkl_str": _hkl_str(A_hkl) if assign_hkl else "—",
            "b_hkl_str": _hkl_str(B_hkl) if assign_hkl else "—",
            "orientation_mode": mode,
            "anchored": bool(anchored),
            "relative_hkl": bool(assign_hkl and not anchored),
        },
        bvm=bvm,
    )


def make_indexing_figure(result: IndexingResult, *, title: str | None = None):
    """Build matplotlib overlay: BVM + hkl labels + g1/g2/a/b arrows."""
    import matplotlib.pyplot as plt

    if result.bvm is None:
        raise ValueError("IndexingResult.bvm is required to draw the overlay")
    bvm = np.asarray(result.bvm, dtype=float)
    disp = np.power(np.clip(bvm, 0.0, 2e3), 0.5)
    origin = result.origin_px

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.imshow(disp, cmap="gray")
    ok = np.array([p.ok for p in result.peaks], dtype=bool)
    qx_abs = np.array([p.qx_abs_px for p in result.peaks])
    qy_abs = np.array([p.qy_abs_px for p in result.peaks])
    ax.scatter(
        qy_abs[ok], qx_abs[ok], s=110, facecolors="none", edgecolors="lime",
        linewidths=1.3, label=f"indexed hkl ({int(ok.sum())})",
    )
    if (~ok).any():
        ax.scatter(
            qy_abs[~ok], qx_abs[~ok], s=60, marker="x", c="red",
            linewidths=1.2, label=f"out of lattice ({int((~ok).sum())})",
        )
    for p in result.peaks:
        if not p.ok:
            continue
        if (p.h, p.k, p.l) == (0, 0, 0):
            continue  # origin or unlabeled (Unknown without relative hkl)
        label = f"({p.h} {p.k} {p.l})"
        ax.annotate(
            label, (p.qy_abs_px, p.qx_abs_px), xytext=(5, -8),
            textcoords="offset points", color="yellow", fontsize=7,
        )

    g1_lab = result.metrics.get("g1_hkl_str") or _hkl_str(result.g1_hkl)
    g2_lab = result.metrics.get("g2_hkl_str") or _hkl_str(result.g2_hkl)
    a_lab = result.metrics.get("a_hkl_str") or _hkl_str(result.a_hkl)
    b_lab = result.metrics.get("b_hkl_str") or _hkl_str(result.b_hkl)
    arrows = (
        (result.g1_px, f"$g_1$ {g1_lab}", "cyan"),
        (result.g2_px, f"$g_2$ {g2_lab}", "orange"),
        (result.basis_a_px, f"$a$ {a_lab}", "magenta"),
        (result.basis_b_px, f"$b$ {b_lab}", "yellow"),
    )
    for gvec, name, color in arrows:
        ax.annotate(
            "",
            xy=(origin[1] + gvec[1], origin[0] + gvec[0]),
            xytext=(origin[1], origin[0]),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        )
        ax.annotate(
            name,
            (origin[1] + gvec[1], origin[0] + gvec[0]),
            xytext=(6, 6), textcoords="offset points",
            color=color, fontsize=10, fontweight="bold",
        )

    za = result.zone_axis
    mode = result.metrics.get("orientation_mode", "known")
    anchored = result.metrics.get("anchored", False)
    mode_tag = f" [{mode}" + (", anchored]" if anchored else "]")
    ax.set_title(
        (title or f"BVM indexed (hkl) — zone axis [{int(za[0])}{int(za[1])}{int(za[2])}]")
        + mode_tag
    )
    ax.set_xlabel("qy (px)")
    ax.set_ylabel("qx (px)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, bvm.shape[1])
    ax.set_ylim(bvm.shape[0], 0)
    fig.tight_layout()
    return fig
