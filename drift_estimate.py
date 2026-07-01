"""fast4d.drift_estimate — cross-scan drift estimation via phase cross-correlation.

Measures the rigid translation (dy, dx) between N strain/ADF maps taken from the
same sample region at different times or sessions.  The result is a per-file CSV
that can be loaded with ``engine.load_drift_csv``.

Scientific context
------------------
When the same area is scanned multiple times, sample drift between acquisitions
causes a small rigid shift between maps.  This module measures those shifts by
cross-correlating each map against a user-chosen reference scan, then writes a
drift CSV so that Fast4D can adjust line-profile / area-ROI positions per file.

This is NOT intra-scan drift (beam/scan distortion within one acquisition).
It is inter-scan drift — the same physical region appearing at slightly different
pixel positions across multiple files.

Algorithm
---------
Phase cross-correlation (sub-pixel, upsampled) from scikit-image.
Falls back to integer-pixel FFT cross-correlation when scikit-image is absent.

Output CSV columns
------------------
stem, shift_dy_px, shift_dx_px
  Compatible with ``engine.load_drift_csv`` (accepts these column names directly).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


# ── phase cross-correlation ────────────────────────────────────────────────────

def _phase_cross_correlation(
    ref: np.ndarray,
    mov: np.ndarray,
    upsample: int = 10,
) -> tuple[float, float, float]:
    """Return (dy, dx, error) — sub-pixel shift of *mov* relative to *ref*.

    Uses scikit-image when available; falls back to integer-pixel FFT otherwise.
    """
    try:
        from skimage.registration import phase_cross_correlation
        shift, error, _ = phase_cross_correlation(
            ref, mov, upsample_factor=upsample, normalization=None
        )
        return float(shift[0]), float(shift[1]), float(error)
    except ImportError:
        pass

    # Integer-pixel fallback via FFT
    F = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    cc = np.real(np.fft.fftshift(np.fft.ifft2(F)))
    idx = np.unravel_index(np.argmax(cc), cc.shape)
    H, W = cc.shape
    dy = int(idx[0]) - H // 2
    dx = int(idx[1]) - W // 2
    return float(dy), float(dx), 0.0


def _crop(img: np.ndarray, roi_bounds: list | None) -> np.ndarray:
    """Crop *img* to ``[x0, x1, y0, y1]`` (cols, rows). None → unchanged.

    A tracking ROI restricts cross-correlation to a feature-rich window so the
    correlation peak stays sharp regardless of how many files are registered.
    """
    if roi_bounds is None or len(roi_bounds) != 4:
        return img
    H, W = img.shape[:2]
    x0, x1, y0, y1 = roi_bounds
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0 = max(0, min(x0, W - 1)); x1 = max(x0 + 1, min(x1, W))
    y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
    return img[y0:y1, x0:x1]


def _prep_image(arr: np.ndarray) -> np.ndarray:
    """Float64 image with NaN replaced by the map mean (for clean FFT)."""
    img = np.asarray(arr, dtype=np.float64)
    bad = ~np.isfinite(img)
    if bad.any():
        img = img.copy()
        fill = float(np.nanmean(img)) if np.isfinite(img).any() else 0.0
        img[bad] = fill
    return img


# ── map extraction from Fast4D Scan objects ───────────────────────────────────

_CH_LABEL = {
    "exx": ("ε_xx", 1),   # strain_raw channel index
    "eyy": ("ε_yy", 0),
    "exy": ("ε_xy", 2),
}

CHANNELS = [
    ("ADF",  "adf"),
    ("ε_xx", "exx"),
    ("ε_yy", "eyy"),
    ("ε_xy", "exy"),
]


def _get_map(scan: Any, key: str, label: str = "without_roi") -> np.ndarray | None:
    """Extract a 2D float array from a Fast4D Scan for cross-correlation."""
    key = key.lower()

    if key == "adf":
        adf = getattr(scan, "adf_cache", None)
        if adf is not None:
            a = np.asarray(adf, dtype=np.float64)
            return a if a.ndim == 2 else None
        return None

    if key in _CH_LABEL:
        _, ch_idx = _CH_LABEL[key]
        sr = getattr(getattr(scan, "state", None), "strain_raw", {}) or {}
        hw3 = sr.get(label)
        if hw3 is None and label == "without_roi":
            hw3 = next(iter(sr.values()), None) if sr else None
        if hw3 is None:
            return None
        arr = np.asarray(hw3, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, ch_idx]
        return None

    return None


# ── main estimation function ──────────────────────────────────────────────────

def estimate_drift(
    scans: list,
    ref_scan: Any,
    *,
    image_key: str = "exx",
    upsample_factor: int = 10,
    max_shift_px: float = 50.0,
    strain_label: str = "without_roi",
    roi_bounds: list | None = None,
    log=None,
) -> list[dict]:
    """Estimate inter-scan drift by cross-correlating maps against *ref_scan*.

    Parameters
    ----------
    scans : list of Scan
        All scans to register (ref_scan should be in this list).
    ref_scan : Scan
        Reference — assigned shift (0, 0).
    image_key : str
        Map channel for cross-correlation: "adf", "exx", "eyy", "exy".
    upsample_factor : int
        Sub-pixel precision = 1/upsample_factor pixels.  10 → 0.1 px.
    max_shift_px : float
        Warn when |shift| exceeds this value.
    strain_label : str
        Which strain map label to use ("without_roi" or "with_roi").
    roi_bounds : [x0, x1, y0, y1], optional
        Tracking ROI — crop every map to this pixel window before
        cross-correlation.  Concentrating registration on a feature-rich
        region (e.g. vacuum + a particle edge) gives a much sharper
        correlation peak and keeps precision stable as more files are added.
        ``None`` → use the whole map.
    log : callable, optional
        Progress callback log(message).

    Returns
    -------
    list of dict with keys:
        name, dy, dx, magnitude, error, is_reference, warning
    """
    def _log(msg):
        if log:
            log(msg)

    ref_img_raw = _get_map(ref_scan, image_key, strain_label)
    if ref_img_raw is None:
        # Try fallback channel order
        for _, alt in CHANNELS:
            if alt != image_key:
                ref_img_raw = _get_map(ref_scan, alt, strain_label)
                if ref_img_raw is not None:
                    _log(f"[Drift] '{image_key}' not in reference; using '{alt}'.")
                    image_key = alt
                    break

    if ref_img_raw is None:
        raise RuntimeError(
            f"Reference scan '{getattr(ref_scan, 'name', '?')}' has no '{image_key}' map. "
            "Compute strain maps first (or load an ADF image)."
        )

    ref_img = _crop(_prep_image(ref_img_raw), roi_bounds)
    roi_note = (f" | tracking ROI {[int(b) for b in roi_bounds]}"
                if roi_bounds else " | whole map")
    _log(f"[Drift] Reference: '{getattr(ref_scan, 'name', '?')}' | "
         f"channel: {image_key} | upsample: {upsample_factor}x{roi_note}")

    results = []
    for sc in scans:
        name = getattr(sc, "name", str(sc))
        is_ref = sc is ref_scan

        if is_ref:
            results.append({"name": name, "dy": 0.0, "dx": 0.0,
                            "magnitude": 0.0, "error": 0.0,
                            "is_reference": True, "warning": ""})
            _log(f"[Drift] '{name}': reference (0.00, 0.00)")
            continue

        mov_raw = _get_map(sc, image_key, strain_label)
        if mov_raw is None:
            _log(f"[Drift] '{name}': no '{image_key}' map — zero shift assumed.")
            results.append({"name": name, "dy": 0.0, "dx": 0.0,
                            "magnitude": 0.0, "error": 1.0,
                            "is_reference": False,
                            "warning": f"No '{image_key}' map found"})
            continue

        mov_img = _crop(_prep_image(mov_raw), roi_bounds)
        if mov_img.shape != ref_img.shape:
            # crop both to the overlapping window so shapes match
            h = min(ref_img.shape[0], mov_img.shape[0])
            w = min(ref_img.shape[1], mov_img.shape[1])
            dy, dx, error = _phase_cross_correlation(
                ref_img[:h, :w], mov_img[:h, :w], upsample_factor)
        else:
            dy, dx, error = _phase_cross_correlation(ref_img, mov_img, upsample_factor)
        mag = float(np.hypot(dy, dx))
        warning = f"Large shift ({mag:.1f} px)" if mag > max_shift_px else ""

        _log(f"[Drift] '{name}': dy={dy:+.2f} dx={dx:+.2f} px  "
             f"error={error:.4f}" + (f"  ⚠ {warning}" if warning else ""))

        results.append({"name": name, "dy": dy, "dx": dx,
                        "magnitude": mag, "error": error,
                        "is_reference": False, "warning": warning})

    return results


# ── CSV export ────────────────────────────────────────────────────────────────

def save_drift_csv(results: list[dict], path: str | Path) -> Path:
    """Write drift results to a CSV compatible with ``engine.load_drift_csv``.

    Columns: stem, shift_dy_px, shift_dx_px
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "shift_dy_px", "shift_dx_px"])
        for r in results:
            w.writerow([r["name"], f"{r['dy']:.4f}", f"{r['dx']:.4f}"])
    return path
