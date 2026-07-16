"""Orientation → peaks motor (py4DSTEM Crystal), UI-free.

Path A (``known_generate``): zone + proj → ``generate_diffraction_pattern`` →
nearest-neighbour match to measured BVM maxima (Å⁻¹ ↔ px via ``Q_pixel``).

Path B (``acom_match``): ``orientation_plan`` + ``match_single_pattern`` →
regenerate theoretical peaks → same matcher.

Does not touch the strain pipeline. Compare against ``bvm_indexing.IndexingResult``
in the GUI only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

Log = Callable[[str], None] | None

# Module-level cache: (crystal_key, voltage, k_max, za_step, ip_step) → Crystal w/ plan
_ORIENTATION_PLAN_CACHE: dict[tuple, Any] = {}


def clear_orientation_plan_cache() -> None:
    """Drop cached ACOM plans (e.g. after CIF change or failed first build)."""
    _ORIENTATION_PLAN_CACHE.clear()


def _log(log: Log, msg: str) -> None:
    if log is not None:
        log(msg)


def _patch_acom_numpy_integer() -> None:
    """py4DSTEM 0.14.19 uses ``.astype(np.integer)`` which NumPy 2.x rejects."""
    import py4DSTEM.process.diffraction.crystal_ACOM as acom

    if getattr(acom, "_fast4d_np_integer_patched", False):
        return

    class _NpProxy:
        integer = np.int64
        signedinteger = np.int64

        def __getattr__(self, name: str):
            return getattr(np, name)

    acom.np = _NpProxy()
    acom._fast4d_np_integer_patched = True


@dataclass
class TheoreticalPeak:
    qx_A: float
    qy_A: float
    intensity: float
    h: int
    k: int
    l: int


@dataclass
class MatchedPeak:
    measured_index: int
    qx_px: float
    qy_px: float
    qx_A: float
    qy_A: float
    intensity: float
    h: int
    k: int
    l: int
    residual_px: float
    residual_A: float
    theo_qx_A: float
    theo_qy_A: float
    theo_intensity: float


@dataclass
class OrientationPeaksResult:
    mode: str  # "known_generate" | "acom_match"
    theoretical_peaks: list[TheoreticalPeak]
    matched: list[MatchedPeak]
    measured_qx_px: np.ndarray
    measured_qy_px: np.ndarray
    measured_intensity: np.ndarray
    measured_qx_abs_px: np.ndarray
    measured_qy_abs_px: np.ndarray
    origin_px: np.ndarray
    Q_pixel: float
    Q_units: str
    tol_px: float
    zone_axis: np.ndarray
    proj_x_lattice: np.ndarray
    index_origin: int
    index_g1: int
    index_g2: int
    g1_px: np.ndarray
    g2_px: np.ndarray
    bvm: np.ndarray | None = None
    orientation: Any = None
    corr_score: float | None = None
    # In-plane rotation that maps theoretical → measured (degrees, Q-space).
    suggested_qr_rotation_deg: float | None = None
    suggested_coordinate_rotation_deg: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    crystal_name: str = ""

    @property
    def n_matched(self) -> int:
        return len(self.matched)

    @property
    def n_theoretical(self) -> int:
        return len(self.theoretical_peaks)

    @property
    def rms_px(self) -> float:
        if not self.matched:
            return float("nan")
        return float(np.sqrt(np.mean([m.residual_px ** 2 for m in self.matched])))


def pointlist_from_maxima_A(
    qx_A: np.ndarray,
    qy_A: np.ndarray,
    intensity: np.ndarray,
):
    """Build an emdfile PointList in Å⁻¹ for ACOM / generate APIs."""
    from emdfile import PointList

    n = int(len(qx_A))
    dtype = np.dtype([("qx", "f8"), ("qy", "f8"), ("intensity", "f8")])
    arr = np.zeros(n, dtype=dtype)
    arr["qx"] = np.asarray(qx_A, dtype=float)
    arr["qy"] = np.asarray(qy_A, dtype=float)
    arr["intensity"] = np.maximum(np.asarray(intensity, dtype=float), 1e-12)
    return PointList(data=arr)


def theoretical_from_pointlist(pl) -> list[TheoreticalPeak]:
    data = getattr(pl, "data", pl)
    out: list[TheoreticalPeak] = []
    for i in range(len(data)):
        row = data[i]
        h = int(row["h"]) if "h" in data.dtype.names else 0
        k = int(row["k"]) if "k" in data.dtype.names else 0
        l = int(row["l"]) if "l" in data.dtype.names else 0
        out.append(
            TheoreticalPeak(
                qx_A=float(row["qx"]),
                qy_A=float(row["qy"]),
                intensity=float(row["intensity"]),
                h=h,
                k=k,
                l=l,
            )
        )
    return out


def match_theoretical_to_measured(
    theo: Sequence[TheoreticalPeak],
    qx_px: np.ndarray,
    qy_px: np.ndarray,
    intensity: np.ndarray,
    *,
    Q_pixel: float,
    tol_px: float,
) -> list[MatchedPeak]:
    """Greedy nearest-neighbour match in Å⁻¹; residual reported in px and Å⁻¹."""
    Q = float(Q_pixel)
    if Q <= 0:
        raise ValueError("Q_pixel must be > 0")
    tol_A = float(tol_px) * Q
    qx_A = np.asarray(qx_px, dtype=float) * Q
    qy_A = np.asarray(qy_px, dtype=float) * Q
    inten = np.asarray(intensity, dtype=float)
    used_meas: set[int] = set()
    used_theo: set[int] = set()
    matches: list[MatchedPeak] = []

    # Prefer strong theoretical peaks first
    order = sorted(range(len(theo)), key=lambda i: theo[i].intensity, reverse=True)
    for ti in order:
        t = theo[ti]
        # Skip (000)
        if abs(t.qx_A) < 1e-9 and abs(t.qy_A) < 1e-9:
            continue
        best_j = -1
        best_d = float("inf")
        for j in range(len(qx_A)):
            if j in used_meas:
                continue
            d = float(np.hypot(qx_A[j] - t.qx_A, qy_A[j] - t.qy_A))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j < 0 or best_d > tol_A:
            continue
        used_meas.add(best_j)
        used_theo.add(ti)
        matches.append(
            MatchedPeak(
                measured_index=int(best_j),
                qx_px=float(qx_px[best_j]),
                qy_px=float(qy_px[best_j]),
                qx_A=float(qx_A[best_j]),
                qy_A=float(qy_A[best_j]),
                intensity=float(inten[best_j]),
                h=int(t.h),
                k=int(t.k),
                l=int(t.l),
                residual_px=float(best_d / Q),
                residual_A=float(best_d),
                theo_qx_A=float(t.qx_A),
                theo_qy_A=float(t.qy_A),
                theo_intensity=float(t.intensity),
            )
        )
    return matches


def estimate_inplane_rotation_deg(
    matched: Sequence[MatchedPeak],
    *,
    orientation: Any = None,
) -> float | None:
    """Estimate Q-plane rotation (deg) that maps theoretical → measured peaks.

    Uses a 2D Procrustes / complex mean of angle differences on matched pairs
    (excludes near-origin). Falls back to ACOM ``Orientation.angles[0, 1]``
    (radians → degrees) when fewer than 2 usable matches.
    """
    vecs_t: list[np.ndarray] = []
    vecs_m: list[np.ndarray] = []
    for m in matched:
        t = np.array([m.theo_qx_A, m.theo_qy_A], dtype=float)
        e = np.array([m.qx_A, m.qy_A], dtype=float)
        if np.hypot(*t) < 1e-6 or np.hypot(*e) < 1e-6:
            continue
        vecs_t.append(t)
        vecs_m.append(e)
    if len(vecs_t) >= 2:
        # Weighted circular mean of atan2 cross/dot
        sins = 0.0
        coss = 0.0
        for t, e in zip(vecs_t, vecs_m):
            # angle of e relative to t
            cross = float(t[0] * e[1] - t[1] * e[0])
            dot = float(t[0] * e[0] + t[1] * e[1])
            w = float(np.hypot(*t) * np.hypot(*e))
            ang = np.arctan2(cross, dot)
            sins += w * np.sin(ang)
            coss += w * np.cos(ang)
        if abs(sins) + abs(coss) > 0:
            deg = float(np.degrees(np.arctan2(sins, coss)))
            # Normalize to (-180, 180]
            deg = ((deg + 180.0) % 360.0) - 180.0
            return deg
    if orientation is not None:
        try:
            ang = np.asarray(orientation.angles, dtype=float).ravel()
            if ang.size >= 2:
                # ACOM convention: angles[:, 1] ≈ in-plane (radians)
                deg = float(np.degrees(ang[1]))
                deg = ((deg + 180.0) % 360.0) - 180.0
                return deg
        except Exception:
            pass
    return None


def _attach_rotation_suggestions(
    result: OrientationPeaksResult,
    *,
    orientation: Any = None,
) -> OrientationPeaksResult:
    theta = estimate_inplane_rotation_deg(result.matched, orientation=orientation)
    result.suggested_qr_rotation_deg = theta
    # Same physical in-plane angle → both Fast4D rotation knobs (user can edit).
    result.suggested_coordinate_rotation_deg = theta
    if theta is not None:
        result.metrics["suggested_qr_rotation_deg"] = float(theta)
        result.metrics["suggested_coordinate_rotation_deg"] = float(theta)
    return result


def propose_indices_from_matches(
    matched: Sequence[MatchedPeak],
    qx_px: np.ndarray,
    qy_px: np.ndarray,
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Pick origin (closest to 0) and two strong non-colinear matched peaks as g1/g2."""
    qx = np.asarray(qx_px, dtype=float)
    qy = np.asarray(qy_px, dtype=float)
    index_origin = int(np.argmin(np.hypot(qx, qy))) if len(qx) else 0

    ranked = sorted(matched, key=lambda m: m.intensity, reverse=True)
    if len(ranked) < 2:
        # Fall back: strongest measured peaks away from origin
        r = np.hypot(qx, qy)
        order = np.argsort(-r)
        g1 = int(order[0]) if len(order) else index_origin
        g2 = int(order[1]) if len(order) > 1 else g1
        return index_origin, g1, g2, np.array([qx[g1], qy[g1]]), np.array([qx[g2], qy[g2]])

    g1_m = ranked[0]
    g1 = int(g1_m.measured_index)
    v1 = np.array([g1_m.qx_px, g1_m.qy_px], dtype=float)
    g2 = g1
    v2 = v1.copy()
    best_score = -1.0
    for m in ranked[1:]:
        v = np.array([m.qx_px, m.qy_px], dtype=float)
        cross = abs(float(v1[0] * v[1] - v1[1] * v[0]))
        if cross > best_score:
            best_score = cross
            g2 = int(m.measured_index)
            v2 = v
    return index_origin, g1, g2, v1, v2


def prepare_crystal_structure_factors(
    crystal,
    *,
    accel_voltage: float = 300_000.0,
    k_max: float = 1.5,
) -> None:
    crystal.setup_diffraction(float(accel_voltage))
    crystal.calculate_structure_factors(k_max=float(k_max))


def k_max_covering_bvm(
    bvm: np.ndarray,
    origin_px: np.ndarray,
    Q_pixel: float,
    *,
    margin: float = 1.02,
) -> float:
    """Smallest k_max (Å⁻¹) that reaches all four corners of the BVM from the origin."""
    bvm = np.asarray(bvm)
    ox, oy = float(origin_px[0]), float(origin_px[1])
    ny, nx = int(bvm.shape[0]), int(bvm.shape[1])
    Q = float(Q_pixel)
    if Q <= 0 or nx < 2 or ny < 2:
        return 1.2
    # Corners in px relative to origin → Å⁻¹
    corners = (
        (0.0 - ox, 0.0 - oy),
        (nx - 1.0 - ox, 0.0 - oy),
        (0.0 - ox, ny - 1.0 - oy),
        (nx - 1.0 - ox, ny - 1.0 - oy),
    )
    r_max = max(float(np.hypot(cx, cy)) for cx, cy in corners)
    return float(r_max * Q * float(margin))


def effective_generation_k_max(
    bvm: np.ndarray,
    origin_px: np.ndarray,
    Q_pixel: float,
    k_max_user: float,
) -> float:
    """Use at least the user k_max, but expand so theory covers the full BVM FOV."""
    k_fov = k_max_covering_bvm(bvm, origin_px, Q_pixel)
    return float(max(float(k_max_user), k_fov))


def generate_theoretical_peaks(
    crystal,
    *,
    zone_axis: Sequence[float] = (0, 0, 1),
    proj_x_lattice: Sequence[float] = (1, 0, 0),
    k_max: float = 1.2,
    orientation: Any = None,
) -> list[TheoreticalPeak]:
    """Generate all theoretical Bragg peaks up to ``k_max`` (Å⁻¹)."""
    kwargs: dict[str, Any] = {"k_max": float(k_max)}
    if orientation is not None:
        kwargs["orientation"] = orientation
    else:
        kwargs["zone_axis_lattice"] = np.asarray(zone_axis, dtype=float).ravel()[:3]
        kwargs["proj_x_lattice"] = np.asarray(proj_x_lattice, dtype=float).ravel()[:3]
    pl = crystal.generate_diffraction_pattern(**kwargs)
    return theoretical_from_pointlist(pl)


def ensure_orientation_plan(
    crystal,
    *,
    crystal_key: str,
    accel_voltage: float = 300_000.0,
    k_max: float = 1.5,
    angle_step_zone_axis: float = 4.0,
    angle_step_in_plane: float = 4.0,
    zone_axis_range: np.ndarray | None = None,
    log: Log = None,
) -> Any:
    """Build / cache ``orientation_plan`` on ``crystal`` (NumPy-2 safe).

    Returns the Crystal that holds the plan (may be a cached instance). Callers
    must use the returned object for ``match_single_pattern``.
    """
    _patch_acom_numpy_integer()
    key = (
        str(crystal_key),
        float(accel_voltage),
        float(k_max),
        float(angle_step_zone_axis),
        float(angle_step_in_plane),
    )
    cached = _ORIENTATION_PLAN_CACHE.get(key)
    if cached is not None and hasattr(cached, "orientation_ref"):
        _log(log, f"Reusing cached orientation_plan for {crystal_key}")
        return cached
    if cached is not None:
        _log(log, f"Cached orientation_plan incomplete for {crystal_key}; rebuilding…")
        _ORIENTATION_PLAN_CACHE.pop(key, None)

    prepare_crystal_structure_factors(crystal, accel_voltage=accel_voltage, k_max=k_max)
    za_range = zone_axis_range
    if za_range is None:
        za_range = np.array([[0, 1, 1], [1, 1, 1]], dtype=float)
    _log(log, f"Building orientation_plan ({crystal_key}, step={angle_step_zone_axis}°)…")
    crystal.orientation_plan(
        zone_axis_range=np.asarray(za_range, dtype=float),
        angle_step_zone_axis=float(angle_step_zone_axis),
        angle_step_in_plane=float(angle_step_in_plane),
        accel_voltage=float(accel_voltage),
        corr_kernel_size=0.08,
        calculate_correlation_array=True,
        progress_bar=False,
    )
    if not hasattr(crystal, "orientation_ref"):
        raise RuntimeError(
            "orientation_plan finished but orientation_ref is missing "
            "(need calculate_correlation_array=True)."
        )
    _ORIENTATION_PLAN_CACHE[key] = crystal
    return crystal


def run_known_generate(
    *,
    crystal,
    crystal_name: str,
    bvm: np.ndarray,
    origin_px: np.ndarray,
    Q_pixel: float,
    Q_units: str = "A^-1",
    zone_axis: Sequence[int] = (1, 1, 0),
    proj_x_lattice: Sequence[int] = (0, 0, -1),
    k_max: float = 1.2,
    tol_px: float = 2.0,
    accel_voltage: float = 300_000.0,
    maxima: dict | None = None,
    min_spacing: float = 20,
    min_absolute_intensity: float = 80,
    max_num_peaks: int = 60,
    edge_boundary: float = 40,
    image_upsample: int = 1,
    log: Log = None,
) -> OrientationPeaksResult:
    """Path A: known orientation → theoretical peaks → match BVM maxima."""
    from plugins.indexing.peaks import find_peaks

    origin_px = np.asarray(origin_px, dtype=float).ravel()[:2]
    k_gen = effective_generation_k_max(bvm, origin_px, Q_pixel, k_max)
    prepare_crystal_structure_factors(
        crystal, accel_voltage=accel_voltage, k_max=max(k_gen, 1.5)
    )
    theo = generate_theoretical_peaks(
        crystal, zone_axis=zone_axis, proj_x_lattice=proj_x_lattice, k_max=k_gen
    )
    _log(
        log,
        f"Path A: {len(theo)} theoretical peaks "
        f"(k_max user={float(k_max):.3g} → FOV-covering {k_gen:.3g} Å⁻¹)",
    )

    if maxima is None:
        maxima = find_peaks(
            bvm,
            min_spacing=min_spacing,
            min_absolute_intensity=min_absolute_intensity,
            max_num_peaks=max_num_peaks,
            edge_boundary=edge_boundary,
            image_upsample=int(image_upsample),
        )
    gx_abs = np.asarray(maxima["x"], dtype=float)
    gy_abs = np.asarray(maxima["y"], dtype=float)
    intensity = np.asarray(maxima["intensity"], dtype=float)
    qx = gx_abs - origin_px[0]
    qy = gy_abs - origin_px[1]

    matched = match_theoretical_to_measured(
        theo, qx, qy, intensity, Q_pixel=Q_pixel, tol_px=tol_px
    )
    i0, i1, i2, g1_px, g2_px = propose_indices_from_matches(matched, qx, qy)
    _log(log, f"Path A: matched {len(matched)}/{len(theo)} within {tol_px} px")

    result = OrientationPeaksResult(
        mode="known_generate",
        theoretical_peaks=theo,
        matched=matched,
        measured_qx_px=qx,
        measured_qy_px=qy,
        measured_intensity=intensity,
        measured_qx_abs_px=gx_abs,
        measured_qy_abs_px=gy_abs,
        origin_px=origin_px.copy(),
        Q_pixel=float(Q_pixel),
        Q_units=str(Q_units),
        tol_px=float(tol_px),
        zone_axis=np.asarray(zone_axis, dtype=int).ravel()[:3],
        proj_x_lattice=np.asarray(proj_x_lattice, dtype=int).ravel()[:3],
        index_origin=i0,
        index_g1=i1,
        index_g2=i2,
        g1_px=g1_px,
        g2_px=g2_px,
        bvm=np.asarray(bvm, dtype=float),
        metrics={
            "n_matched": len(matched),
            "n_theoretical": len(theo),
            "n_measured": len(qx),
            "rms_px": float(np.sqrt(np.mean([m.residual_px ** 2 for m in matched]))) if matched else float("nan"),
            "k_max": float(k_max),
            "k_max_generated": float(k_gen),
        },
        crystal_name=str(crystal_name),
    )
    result = _attach_rotation_suggestions(result)
    if result.suggested_qr_rotation_deg is not None:
        _log(log, f"Path A: suggested QR/coord rotation ≈ {result.suggested_qr_rotation_deg:.2f}°")
    return result


def run_acom_match(
    *,
    crystal,
    crystal_key: str,
    crystal_name: str,
    bvm: np.ndarray,
    origin_px: np.ndarray,
    Q_pixel: float,
    Q_units: str = "A^-1",
    k_max: float = 1.2,
    tol_px: float = 2.0,
    accel_voltage: float = 300_000.0,
    angle_step_zone_axis: float = 4.0,
    angle_step_in_plane: float = 4.0,
    maxima: dict | None = None,
    min_spacing: float = 20,
    min_absolute_intensity: float = 80,
    max_num_peaks: int = 60,
    edge_boundary: float = 40,
    image_upsample: int = 1,
    log: Log = None,
) -> OrientationPeaksResult:
    """Path B: ACOM match_single_pattern → regenerate → match."""
    from plugins.indexing.peaks import find_peaks

    crystal = ensure_orientation_plan(
        crystal,
        crystal_key=crystal_key,
        accel_voltage=accel_voltage,
        k_max=max(k_max, 1.5),
        angle_step_zone_axis=angle_step_zone_axis,
        angle_step_in_plane=angle_step_in_plane,
        log=log,
    )

    origin_px = np.asarray(origin_px, dtype=float).ravel()[:2]
    k_gen = effective_generation_k_max(bvm, origin_px, Q_pixel, k_max)
    if maxima is None:
        maxima = find_peaks(
            bvm,
            min_spacing=min_spacing,
            min_absolute_intensity=min_absolute_intensity,
            max_num_peaks=max_num_peaks,
            edge_boundary=edge_boundary,
            image_upsample=int(image_upsample),
        )
    gx_abs = np.asarray(maxima["x"], dtype=float)
    gy_abs = np.asarray(maxima["y"], dtype=float)
    intensity = np.asarray(maxima["intensity"], dtype=float)
    qx = gx_abs - origin_px[0]
    qy = gy_abs - origin_px[1]
    Q = float(Q_pixel)
    pl = pointlist_from_maxima_A(qx * Q, qy * Q, intensity)

    _log(log, f"Path B: match_single_pattern ({len(pl.data)} peaks)…")
    orient = crystal.match_single_pattern(
        pl,
        num_matches_return=1,
        plot_polar=False,
        plot_corr=False,
        verbose=False,
    )
    corr = None
    try:
        corr = float(np.asarray(orient.corr).ravel()[0])
    except Exception:
        corr = None

    # Expand structure factors so regenerated pattern fills the BVM FOV
    # (orientation_plan may have used a smaller k_max for ACOM).
    prepare_crystal_structure_factors(
        crystal, accel_voltage=accel_voltage, k_max=max(k_gen, 1.5)
    )
    theo = generate_theoretical_peaks(
        crystal,
        zone_axis=(0, 0, 1),
        proj_x_lattice=(1, 0, 0),
        k_max=k_gen,
        orientation=orient,
    )
    matched = match_theoretical_to_measured(
        theo, qx, qy, intensity, Q_pixel=Q_pixel, tol_px=tol_px
    )
    i0, i1, i2, g1_px, g2_px = propose_indices_from_matches(matched, qx, qy)
    _log(
        log,
        f"Path B: corr={corr} matched {len(matched)}/{len(theo)} "
        f"(k_max FOV={k_gen:.3g} Å⁻¹)",
    )

    # Best-effort zone from orientation matrix (crystal → sample)
    zone = np.array([0, 0, 1], dtype=int)
    proj = np.array([1, 0, 0], dtype=int)
    try:
        M = np.asarray(orient.matrix)
        if M.ndim == 3:
            M = M[0]
        # Last column / row heuristics vary; store matrix in metrics
        zone = np.round(M[:, 2] if M.shape == (3, 3) else [0, 0, 1]).astype(int)
    except Exception:
        pass

    result = OrientationPeaksResult(
        mode="acom_match",
        theoretical_peaks=theo,
        matched=matched,
        measured_qx_px=qx,
        measured_qy_px=qy,
        measured_intensity=intensity,
        measured_qx_abs_px=gx_abs,
        measured_qy_abs_px=gy_abs,
        origin_px=origin_px.copy(),
        Q_pixel=float(Q_pixel),
        Q_units=str(Q_units),
        tol_px=float(tol_px),
        zone_axis=zone.ravel()[:3],
        proj_x_lattice=proj.ravel()[:3],
        index_origin=i0,
        index_g1=i1,
        index_g2=i2,
        g1_px=g1_px,
        g2_px=g2_px,
        bvm=np.asarray(bvm, dtype=float),
        orientation=orient,
        corr_score=corr,
        metrics={
            "n_matched": len(matched),
            "n_theoretical": len(theo),
            "n_measured": len(qx),
            "rms_px": float(np.sqrt(np.mean([m.residual_px ** 2 for m in matched]))) if matched else float("nan"),
            "corr_score": corr,
            "k_max": float(k_max),
            "k_max_generated": float(k_gen),
            "angle_step_zone_axis": float(angle_step_zone_axis),
        },
        crystal_name=str(crystal_name),
    )
    result = _attach_rotation_suggestions(result, orientation=orient)
    if result.suggested_qr_rotation_deg is not None:
        _log(log, f"Path B: suggested QR/coord rotation ≈ {result.suggested_qr_rotation_deg:.2f}°")
    return result


def compare_to_indexing_result(
    orient_result: OrientationPeaksResult,
    indexing_result: Any,
) -> dict[str, Any]:
    """Lightweight comparison vs our Index BVM result (no writes)."""
    out: dict[str, Any] = {
        "orient_mode": orient_result.mode,
        "orient_n_matched": orient_result.n_matched,
        "orient_rms_px": orient_result.rms_px,
        "orient_g1": orient_result.index_g1,
        "orient_g2": orient_result.index_g2,
        "orient_origin": orient_result.index_origin,
    }
    if indexing_result is None:
        out["indexing_available"] = False
        return out
    out["indexing_available"] = True
    out["index_g1"] = int(indexing_result.index_g1)
    out["index_g2"] = int(indexing_result.index_g2)
    out["index_origin"] = int(indexing_result.index_origin)
    out["index_n_inliers"] = int(getattr(indexing_result, "n_inliers", 0))
    out["same_g1"] = out["orient_g1"] == out["index_g1"]
    out["same_g2"] = out["orient_g2"] == out["index_g2"]
    out["same_origin"] = out["orient_origin"] == out["index_origin"]
    # hkl overlap by measured peak index
    ours = {
        int(p.peak_index): (int(p.h), int(p.k), int(p.l))
        for p in getattr(indexing_result, "peaks", [])
        if getattr(p, "ok", False)
    }
    py = {
        int(m.measured_index): (int(m.h), int(m.k), int(m.l))
        for m in orient_result.matched
    }
    common = set(ours) & set(py)
    agree = sum(1 for i in common if ours[i] == py[i] or ours[i] == tuple(-x for x in py[i]))
    out["hkl_common_peaks"] = len(common)
    out["hkl_agree"] = agree
    return out


def _hkl_label(h: int, k: int, l: int) -> str:
    return f"({int(h)} {int(k)} {int(l)})"


def make_orientation_peaks_figure(
    result: OrientationPeaksResult,
    *,
    title: str | None = None,
    indexing_result: Any = None,
    label_theoretical: bool = True,
    max_theo_labels: int = 40,
):
    """2×2 figure: Theoretical | Measured | Matched | Matched+g1/g2."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    ax_theo, ax_meas = axes[0, 0], axes[0, 1]
    ax_match, ax_g = axes[1, 0], axes[1, 1]

    bvm = result.bvm
    ox, oy = float(result.origin_px[0]), float(result.origin_px[1])
    Q = float(result.Q_pixel)
    matched_by_idx = {int(m.measured_index): m for m in result.matched}

    def _show_bvm(ax) -> None:
        if bvm is not None:
            ax.imshow(bvm, cmap="gray", origin="lower")
        ax.set_aspect("equal")
        ax.set_xlabel("qx (px)")
        ax.set_ylabel("qy (px)")

    # ── 1 Theoretical ──────────────────────────────────────────────────────
    _show_bvm(ax_theo)
    ax_theo.set_title("Theoretical (CIF)")
    if result.theoretical_peaks:
        tx = np.array([t.qx_A / Q + ox for t in result.theoretical_peaks])
        ty = np.array([t.qy_A / Q + oy for t in result.theoretical_peaks])
        ax_theo.scatter(tx, ty, s=40, facecolors="none", edgecolors="#FF9800", linewidths=1.2)
        if label_theoretical:
            ranked = sorted(result.theoretical_peaks, key=lambda t: t.intensity, reverse=True)
            n_lab = 0
            for t in ranked:
                if (t.h, t.k, t.l) == (0, 0, 0):
                    continue
                if n_lab >= int(max_theo_labels):
                    break
                ax_theo.annotate(
                    _hkl_label(t.h, t.k, t.l),
                    (t.qx_A / Q + ox, t.qy_A / Q + oy),
                    xytext=(3, 3), textcoords="offset points",
                    color="#FFB74D", fontsize=6, alpha=0.95,
                )
                n_lab += 1

    # ── 2 Experimental / Measured ──────────────────────────────────────────
    _show_bvm(ax_meas)
    ax_meas.set_title("Experimental / Measured")
    ax_meas.scatter(
        result.measured_qx_abs_px, result.measured_qy_abs_px,
        s=28, c="#42A5F5", alpha=0.85,
    )
    if 0 <= result.index_origin < len(result.measured_qx_abs_px):
        ax_meas.scatter(
            [result.measured_qx_abs_px[result.index_origin]],
            [result.measured_qy_abs_px[result.index_origin]],
            s=80, facecolors="none", edgecolors="#FFEB3B", linewidths=1.8,
        )

    # ── 3 Matched ──────────────────────────────────────────────────────────
    _show_bvm(ax_match)
    ax_match.set_title(f"Matched ({result.n_matched})")
    if result.theoretical_peaks:
        tx = np.array([t.qx_A / Q + ox for t in result.theoretical_peaks])
        ty = np.array([t.qy_A / Q + oy for t in result.theoretical_peaks])
        ax_match.scatter(tx, ty, s=30, facecolors="none", edgecolors="#FF9800", linewidths=1.0, alpha=0.45)
    for m in result.matched:
        x, y = m.qx_px + ox, m.qy_px + oy
        ax_match.scatter([x], [y], s=40, c="#00E676", marker="x", linewidths=1.5)
        if (m.h, m.k, m.l) != (0, 0, 0):
            ax_match.annotate(
                _hkl_label(m.h, m.k, m.l),
                (x, y), xytext=(4, -8), textcoords="offset points",
                color="#69F0AE", fontsize=7, fontweight="bold",
            )

    # ── 4 Matched + g1/g2 ──────────────────────────────────────────────────
    _show_bvm(ax_g)
    rot = result.suggested_qr_rotation_deg
    rot_txt = f"  QR≈{rot:.1f}°" if rot is not None else ""
    ax_g.set_title(f"Matched + g1/g2{rot_txt}")
    for m in result.matched:
        x, y = m.qx_px + ox, m.qy_px + oy
        ax_g.scatter([x], [y], s=36, c="#00E676", marker="x", linewidths=1.3)
        if (m.h, m.k, m.l) != (0, 0, 0):
            ax_g.annotate(
                _hkl_label(m.h, m.k, m.l),
                (x, y), xytext=(4, -8), textcoords="offset points",
                color="#69F0AE", fontsize=6,
            )
    for idx, color, lab in (
        (result.index_g1, "#E53935", "g1"),
        (result.index_g2, "#8E24AA", "g2"),
        (result.index_origin, "#FFEB3B", "origin"),
    ):
        if not (0 <= idx < len(result.measured_qx_abs_px)):
            continue
        x = float(result.measured_qx_abs_px[idx])
        y = float(result.measured_qy_abs_px[idx])
        ax_g.scatter([x], [y], s=110, facecolors="none", edgecolors=color, linewidths=2.2)
        m = matched_by_idx.get(int(idx))
        if m is not None and (m.h, m.k, m.l) != (0, 0, 0):
            text = f"{lab} {_hkl_label(m.h, m.k, m.l)}"
        elif lab == "origin":
            text = f"{lab} (0 0 0)"
        else:
            text = lab
        ax_g.annotate(
            text, (x, y), xytext=(6, 6), textcoords="offset points",
            color=color, fontsize=8, fontweight="bold",
        )
        # Arrow from origin to g1/g2
        if lab in ("g1", "g2") and 0 <= result.index_origin < len(result.measured_qx_abs_px):
            x0 = float(result.measured_qx_abs_px[result.index_origin])
            y0 = float(result.measured_qy_abs_px[result.index_origin])
            ax_g.annotate(
                "", xy=(x, y), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8),
            )

    if indexing_result is not None:
        peaks = getattr(indexing_result, "peaks", [])
        for idx, color, lab in (
            (int(indexing_result.index_g1), "#EF9A9A", "Idx g1"),
            (int(indexing_result.index_g2), "#CE93D8", "Idx g2"),
        ):
            pk = next((p for p in peaks if int(p.peak_index) == idx), None)
            if pk is None:
                continue
            ax_g.scatter(
                [pk.qx_abs_px], [pk.qy_abs_px],
                s=80, marker="s", facecolors="none", edgecolors=color, linewidths=1.5,
            )
            hkl = (
                _hkl_label(pk.h, pk.k, pk.l)
                if getattr(pk, "ok", False) and (pk.h, pk.k, pk.l) != (0, 0, 0)
                else ""
            )
            ax_g.annotate(
                f"{lab} {hkl}".strip(),
                (pk.qx_abs_px, pk.qy_abs_px),
                xytext=(6, -12), textcoords="offset points",
                color=color, fontsize=7,
            )

    mode = result.mode
    ttl = title or f"Orient. peaks [{mode}] — {result.crystal_name}"
    if result.corr_score is not None:
        ttl += f"  corr={result.corr_score:.3f}"
    if rot is not None:
        ttl += f"  |  suggested QR = coord = {rot:.2f}°"
    fig.suptitle(ttl, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def export_matches_csv(result: OrientationPeaksResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["measured_index,h,k,l,qx_px,qy_px,residual_px,intensity,theo_qx_A,theo_qy_A"]
    for m in result.matched:
        lines.append(
            f"{m.measured_index},{m.h},{m.k},{m.l},{m.qx_px:.6f},{m.qy_px:.6f},"
            f"{m.residual_px:.6f},{m.intensity:.6f},{m.theo_qx_A:.6f},{m.theo_qy_A:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
