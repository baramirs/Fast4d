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
