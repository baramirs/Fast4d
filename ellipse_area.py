"""ellipse_area -- Qt-free numbers of the "Ellipse from area" tool.

The detector ellipse is fitted on a diffraction image with isotropic rings: an amorphous halo
or polycrystalline rings averaged over a user-drawn scan area. Averaging raw patterns smears
the ring when the beam moves across the scan (descan), so every pattern is shifted to a common
beam centre first. Pattern axes follow py4DSTEM: qx = row, qy = column.

``data`` is anything indexable as ``data[rx, ry] -> (Qx, Qy)`` (ndarray, memmap, DataCube.data);
patterns are read one at a time, so a memory-mapped .mib is never loaded whole.
"""
from __future__ import annotations

import numpy as np


def _positions(mask) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        raise ValueError("the scan area is empty: draw at least one area over the sample")
    return np.argwhere(m)


def com_center(img, guess, radius: float = 10.0, n_iter: int = 3) -> tuple[float, float]:
    """Centre of mass of the direct beam: pixels within ``radius`` of the current estimate and
    above half the local maximum, iterated from ``guess`` (qx, qy)."""
    a = np.asarray(img, dtype=float)
    qx, qy = np.meshgrid(np.arange(a.shape[0], dtype=float), np.arange(a.shape[1], dtype=float),
                         indexing="ij")
    cx, cy = float(guess[0]), float(guess[1])
    for _ in range(int(n_iter)):
        win = np.hypot(qx - cx, qy - cy) <= radius
        if not win.any():
            break
        v = a[win]
        w = np.clip(v - 0.5 * v.max(), 0.0, None)
        if w.sum() <= 0:
            break
        cx = float((qx[win] * w).sum() / w.sum())
        cy = float((qy[win] * w).sum() / w.sum())
    return cx, cy


def beam_centers_com(data, mask, guess, radius: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """(qx0, qy0) maps of the direct-beam centre at every masked scan position (NaN elsewhere)."""
    pos = _positions(mask)
    shape = np.asarray(mask).shape
    qx0 = np.full(shape, np.nan)
    qy0 = np.full(shape, np.nan)
    for i, j in pos:
        qx0[i, j], qy0[i, j] = com_center(np.asarray(data[i, j], dtype=float), guess, radius)
    return qx0, qy0


FIT_METHODS = ("amorphous", "ring")


def fit_ellipse(image, center, r_range, method: str = "amorphous") -> dict:
    """Fit the detector ellipse on a ring of ``image`` between radii ``r_range`` (px).

    method "amorphous": py4DSTEM fit_ellipse_amorphous_ring (broad halo, double-sided
    Gaussian); "ring": py4DSTEM fit_ellipse_1D (sharp polycrystalline ring).
    Returns {p_ellipse (qx0, qy0, a, b, theta_rad), a_over_b (>= 1), theta_deg (angle of the
    MAJOR axis from +qx towards +qy, in [0, 180)), method, r_range, center}."""
    from py4DSTEM.process.calibration import fit_ellipse_1D, fit_ellipse_amorphous_ring
    m = str(method).lower()
    if m not in FIT_METHODS:
        raise ValueError(f"ellipse fit method must be one of {FIT_METHODS}, got {method!r}")
    img = np.asarray(image, dtype=float)
    r0, r1 = sorted(float(v) for v in r_range)
    if not (0 <= r0 < r1):
        raise ValueError(f"ring radii must satisfy 0 <= r0 < r1, got {r_range!r}")
    c = (float(center[0]), float(center[1]))
    if m == "amorphous":
        out = fit_ellipse_amorphous_ring(img, center=c, fitradii=(r0, r1))
        p = out[0] if isinstance(out, tuple) else out
    else:
        p = fit_ellipse_1D(img, center=c, fitradii=(r0, r1))
    p = tuple(float(v) for v in np.ravel(p)[:5])
    a, b, th = abs(p[2]), abs(p[3]), p[4]
    major = th if a >= b else th + np.pi / 2.0
    return dict(p_ellipse=p, a_over_b=float(max(a, b) / min(a, b)),
                theta_deg=float(np.degrees(major) % 180.0), method=m, r_range=(r0, r1),
                center=c)


def ellipse_matrix(p_ellipse) -> np.ndarray:
    """Area-normalised 2x2 distortion R(theta) diag(a, b) R(-theta) of an ellipse
    (qx0, qy0, a, b, theta) or (a, b, theta); None = identity."""
    if p_ellipse is None:
        return np.eye(2)
    p = list(p_ellipse)[-3:]
    a, b, t = abs(float(p[0])), abs(float(p[1])), float(p[2])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return R @ np.diag([a, b]) @ R.T / np.sqrt(a * b)


def residual_ellipse(p_measured, p_applied) -> dict:
    """Ellipticity left on a pattern whose measured ellipse is ``p_measured`` after the
    correction of ``p_applied``: D_res = D_measured D_applied^-1. Returns {a_over_b,
    theta_deg (major axis, [0, 180))}; a_over_b = 1 when the applied ellipse matches."""
    Dr = ellipse_matrix(p_measured) @ np.linalg.inv(ellipse_matrix(p_applied))
    U, S, _ = np.linalg.svd(Dr)
    return dict(a_over_b=float(S[0] / S[1]),
                theta_deg=float(np.degrees(np.arctan2(U[1, 0], U[0, 0])) % 180.0))


def mean_dp(data, mask, origins=None, recenter: bool = True) -> dict:
    """Mean diffraction pattern over the masked scan positions.

    ``origins`` = (qx0, qy0) maps of the beam centre (scan-shaped). With ``recenter`` every
    pattern is shifted (bilinear, sub-pixel) so its beam lands on the mean centre of the area
    before it is added. Returns {image, center (qx, qy) or None, n_patterns, recentered}."""
    from scipy.ndimage import shift as _shift
    pos = _positions(mask)
    if recenter and origins is None:
        raise ValueError("recentering needs the beam centre of every pattern (origins)")
    center = None
    if origins is not None:
        qx0 = np.asarray(origins[0], dtype=float)
        qy0 = np.asarray(origins[1], dtype=float)
        m = np.asarray(mask, dtype=bool)
        center = (float(np.nanmean(qx0[m])), float(np.nanmean(qy0[m])))
    acc = None
    for i, j in pos:
        p = np.asarray(data[i, j], dtype=np.float64)
        if recenter:
            p = _shift(p, (center[0] - qx0[i, j], center[1] - qy0[i, j]), order=1, mode="nearest")
        acc = p.copy() if acc is None else acc + p
    return dict(image=acc / len(pos), center=center, n_patterns=int(len(pos)),
                recentered=bool(recenter))
