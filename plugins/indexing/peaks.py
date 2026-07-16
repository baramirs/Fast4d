"""Shared BVM peak finding with optional spatial upsampling (sampling).

``image_upsample`` (1|2|4) zooms the BVM before ``get_maxima_2D``, then scales
coordinates back to the original pixel grid (sub-pixel). This is distinct from
py4DSTEM's multicorr ``upsample_factor`` (Fourier peak refinement).
"""
from __future__ import annotations

import numpy as np


def _structured_maxima(x: np.ndarray, y: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    n = int(len(x))
    dtype = np.dtype([("x", "f8"), ("y", "f8"), ("intensity", "f8")])
    out = np.zeros(n, dtype=dtype)
    out["x"] = np.asarray(x, dtype=float)
    out["y"] = np.asarray(y, dtype=float)
    out["intensity"] = np.asarray(intensity, dtype=float)
    # Intensity descending (same contract as find_bvm_maxima / choose_basis_vectors)
    order = np.argsort(-out["intensity"])
    return out[order]


def find_peaks(
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
    """Detect maxima on a BVM; optional spatial ``image_upsample`` then scale back.

    Returns a structured array with fields ``x``, ``y``, ``intensity``, sorted by
    intensity descending — the index order that ``index_origin/g1/g2`` refer to.
    """
    from py4DSTEM.preprocess.utils import get_maxima_2D

    arr = np.asarray(bvm, dtype=float)
    factor = int(image_upsample) if image_upsample else 1
    if factor < 1:
        factor = 1
    if factor not in (1, 2, 4, 8):
        # Allow other integers but clamp wild values
        factor = max(1, min(int(factor), 8))

    work = arr
    scale_spacing = 1.0
    scale_edge = 1.0
    if factor > 1:
        try:
            from scipy.ndimage import zoom
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "scipy is required for image_upsample > 1 (BVM spatial sampling)"
            ) from exc
        work = zoom(arr, float(factor), order=3)
        scale_spacing = float(factor)
        scale_edge = float(factor)

    maxima = get_maxima_2D(
        work,
        subpixel=subpixel,
        upsample_factor=int(upsample_factor),
        sigma=float(sigma),
        minAbsoluteIntensity=float(min_absolute_intensity),
        minRelativeIntensity=0,
        relativeToPeak=0,
        minSpacing=float(min_spacing) * scale_spacing,
        edgeBoundary=int(round(float(edge_boundary) * scale_edge)),
        maxNumPeaks=int(max_num_peaks),
    )

    x = np.asarray(maxima["x"], dtype=float)
    y = np.asarray(maxima["y"], dtype=float)
    inten = np.asarray(maxima["intensity"], dtype=float)
    if factor > 1:
        x = x / float(factor)
        y = y / float(factor)
    return _structured_maxima(x, y, inten)
