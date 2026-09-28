"""fast4d.drift_estimate — inter-scan drift from the ADF / BF (or strain) maps.

Measures the rigid translation (dy, dx) between N scans of the SAME sample
region, acquired one after the other under the same conditions (e.g. three
repeats for a reproducibility test). The per-file CSV it writes is read by
``engine.load_drift_csv`` and used by the line / area-ROI plugin to place the
same line on the same physical feature in every file.

This is NOT intra-scan drift (distortion within one acquisition).

Algorithm
---------
* Map per scan: ADF, BF, ADF+BF, or a strain component. ADF and BF come from
  the scan's virtual images (memory, else the light .h5, else the saved
  ``virtual_images/*.npy``) — no 4D datacube is needed.
* Pre-processing: NaN fill, subtraction of a smooth background (removes
  thickness / illumination gradients that would pull the peak to zero shift)
  and a Tukey window (removes the edge discontinuity of the FFT).
* Phase cross-correlation, sub-pixel (scikit-image, upsampled); integer FFT
  fallback when scikit-image is missing.
* ``ADF+BF``: the shift is measured on each image separately and averaged.
  The two are independent contrasts of the same field, so their difference
  is an honest uncertainty of the shift (reported as ``adf_bf_diff_px``).
* Mode ``reference``: every scan against one reference scan.
  Mode ``chained``: every scan against the previous one in the chosen order,
  shifts accumulated. Use it when the contrast changes along the series
  (contamination, beam damage) so that the first and last scans no longer
  correlate well; its error grows with the number of links.
* Quality: normalised cross-correlation (NCC) of the overlapping area before
  and after applying the shift. A good registration raises NCC close to the
  reference-vs-itself level; an NCC after < 0.5 is flagged.

Sign convention (verified by ``tests/test_drift_estimate.py``)
-------------------------------------------------------------
``dy, dx`` is the shift that REGISTERS the scan onto the reference
(scikit-image convention): if a feature sits at +3 px (rows) in the scan
relative to the reference, dy = -3. ``engine._line_drift_shift`` negates it,
so template lines move +3 px and land on the same feature.

Output CSV columns
------------------
stem, shift_dy_px, shift_dx_px            (read by engine.load_drift_csv)
+ channel, mode, reference, ncc_before, ncc_after, adf_bf_diff_px,
  shift_um-free extras that the loader ignores.
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
    """Return (dy, dx, error) — sub-pixel shift that registers *mov* onto *ref*.

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


def _tukey2d(shape, alpha: float = 0.25) -> np.ndarray:
    def tk(n):
        if n < 3 or alpha <= 0:
            return np.ones(n)
        x = np.linspace(0.0, 1.0, n)
        w = np.ones(n)
        a = alpha / 2.0
        lo, hi = x < a, x > 1 - a
        w[lo] = 0.5 * (1 + np.cos(np.pi * (x[lo] / a - 1)))
        w[hi] = 0.5 * (1 + np.cos(np.pi * ((x[hi] - 1) / a + 1)))
        return w
    return np.outer(tk(shape[0]), tk(shape[1]))


def condition(img: np.ndarray, *, background_sigma: float = 0.0,
              window: bool = True) -> np.ndarray:
    """Image ready for phase correlation: background removed, zero mean,
    unit std, tapered edges. ``background_sigma`` (px) = Gaussian width of the
    smooth background subtracted; 0 → auto (1/8 of the smaller side)."""
    from scipy.ndimage import gaussian_filter
    a = _prep_image(img)
    s = background_sigma if background_sigma > 0 else max(2.0, min(a.shape) / 8.0)
    a = a - gaussian_filter(a, s)
    a = a - a.mean()
    sd = a.std()
    if sd > 0:
        a = a / sd
    if window:
        a = a * _tukey2d(a.shape)
    return a


# ── quality ───────────────────────────────────────────────────────────────────

def shift_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Apply the registration shift (sub-pixel, NaN outside)."""
    from scipy.ndimage import shift as _shift
    a = _prep_image(img)
    out = _shift(a, (dy, dx), order=1, mode="constant", cval=np.nan)
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation over the pixels finite in both."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 16:
        return float("nan")
    x = a[m] - a[m].mean()
    y = b[m] - b[m].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


# ── map extraction from Fast4D Scan objects ───────────────────────────────────

_CH_LABEL = {
    "exx": ("ε_xx", 1),   # strain_raw channel index
    "eyy": ("ε_yy", 0),
    "exy": ("ε_xy", 2),
}

CHANNELS = [
    ("ADF + BF", "adf+bf"),
    ("ADF",  "adf"),
    ("BF",   "bf"),
    ("ε_xx", "exx"),
    ("ε_yy", "eyy"),
    ("ε_xy", "exy"),
]

_VKEY = {"adf": "annular_dark_field", "bf": "bright_field"}


def _as2d(node) -> np.ndarray | None:
    if node is None:
        return None
    arr = np.asarray(getattr(node, "data", node), dtype=np.float64)
    return arr if arr.ndim == 2 else None


def _virtual(scan: Any, key: str) -> np.ndarray | None:
    """ADF / BF of a scan without touching the 4D datacube: memory first, then
    the light .h5 the scan points to, then the saved virtual_images/*.npy."""
    vkey = _VKEY[key]
    if key == "adf":
        a = _as2d(getattr(scan, "adf_cache", None))
        if a is not None:
            return a
    st = getattr(scan, "state", None)
    vi = (getattr(st, "virtual_images", None) or {}) if st is not None else {}
    a = _as2d(vi.get(vkey))
    if a is not None:
        return a
    # light h5 (braggpeaks / sidecar) — same reader the ADF preview uses
    for src in (getattr(scan, "h5_path", None), getattr(scan, "braggpeaks_path", None),
                getattr(scan, "raw_path", None)):
        if not src or Path(src).suffix.lower() not in (".h5", ".hdf5", ".emd"):
            continue
        if not Path(src).is_file():
            continue
        try:
            import engine as _E
            imgs = _E._read_virtual_images_lenient(str(src))
        except Exception:
            imgs = {}
        a = _as2d(imgs.get(vkey))
        if a is not None:
            if st is not None:
                vi = dict(vi); vi.update({k: v for k, v in imgs.items() if v is not None})
                try:
                    st.virtual_images = vi
                except Exception:
                    pass
            return a
    # saved results: <results_dir>/virtual_images/*<ADF|BF>*.npy
    rd = getattr(scan, "results_dir", None)
    if rd:
        d = Path(rd) / "virtual_images"
        tag = "ADF" if key == "adf" else "BF"
        for p in sorted(d.glob("*.npy")) if d.is_dir() else []:
            stem = p.stem.upper()
            if tag in stem.split("_") or stem.endswith(tag) or vkey.upper() in stem:
                try:
                    return _as2d(np.load(p))
                except Exception:
                    pass
    return None


def _get_map(scan: Any, key: str, label: str = "without_roi") -> np.ndarray | None:
    """Extract a 2D float array from a Fast4D Scan for cross-correlation."""
    key = key.lower()

    if key in ("adf", "bf"):
        return _virtual(scan, key)

    if key in _CH_LABEL:
        _, ch_idx = _CH_LABEL[key]
        sr = getattr(getattr(scan, "state", None), "strain_raw", {}) or {}
        hw3 = sr.get(label)
        if hw3 is None and label == "without_roi":
            hw3 = next(iter(sr.values()), None) if sr else None
        if hw3 is None:
            return None
        # live state holds (5,H,W) channels-first, a reloaded workspace (H,W,3|5):
        # _as_hw3 normalises both (indexing [:, :, idx] on (5,H,W) read a wrong slice)
        try:
            from fast_artifacts import _as_hw3
        except ImportError:
            from .fast_artifacts import _as_hw3
        arr = _as_hw3(hw3)
        if arr is not None:
            return np.asarray(arr[:, :, ch_idx], dtype=np.float64)
        return None

    return None


def available_channels(scan: Any, label: str = "without_roi") -> list[str]:
    """Channel keys that exist for this scan (for the dialog)."""
    out = [k for k in ("adf", "bf", "exx", "eyy", "exy") if _get_map(scan, k, label) is not None]
    if "adf" in out and "bf" in out:
        out.insert(0, "adf+bf")
    return out


# ── one pair ──────────────────────────────────────────────────────────────────

def _register_pair(ref_raw, mov_raw, roi_bounds, upsample, bg_sigma, window):
    """(dy, dx, error, ncc_before, ncc_after) for one channel."""
    ref_c = _crop(_prep_image(ref_raw), roi_bounds)
    mov_c = _crop(_prep_image(mov_raw), roi_bounds)
    h = min(ref_c.shape[0], mov_c.shape[0])
    w = min(ref_c.shape[1], mov_c.shape[1])
    ref_c, mov_c = ref_c[:h, :w], mov_c[:h, :w]
    dy, dx, err = _phase_cross_correlation(
        condition(ref_c, background_sigma=bg_sigma, window=window),
        condition(mov_c, background_sigma=bg_sigma, window=window), upsample)
    # quality on the whole map (not only the ROI) with the smooth background
    # removed, so a shared illumination gradient does not inflate NCC
    ref_q = condition(_prep_image(ref_raw), background_sigma=bg_sigma, window=False)
    mov_q = condition(_prep_image(mov_raw), background_sigma=bg_sigma, window=False)
    if ref_q.shape != mov_q.shape:
        hh = min(ref_q.shape[0], mov_q.shape[0]); ww = min(ref_q.shape[1], mov_q.shape[1])
        ref_q, mov_q = ref_q[:hh, :ww], mov_q[:hh, :ww]
    before = ncc(ref_q, mov_q)
    after = ncc(ref_q, shift_image(mov_q, dy, dx))
    return dy, dx, err, before, after


def _pair(ref_scan, mov_scan, key, label, roi_bounds, upsample, bg_sigma, window):
    """Register one scan onto another with channel *key* (``adf+bf`` averages)."""
    keys = ("adf", "bf") if key == "adf+bf" else (key,)
    per = {}
    for k in keys:
        r = _get_map(ref_scan, k, label)
        m = _get_map(mov_scan, k, label)
        if r is None or m is None:
            continue
        per[k] = _register_pair(r, m, roi_bounds, upsample, bg_sigma, window)
    if not per:
        return None
    arr = np.array([v for v in per.values()], float)
    dy, dx, err, before, after = arr.mean(axis=0)
    diff = float(np.hypot(*(arr[0, :2] - arr[1, :2]))) if len(per) == 2 else float("nan")
    return dict(dy=float(dy), dx=float(dx), error=float(err), ncc_before=float(before),
                ncc_after=float(after), adf_bf_diff_px=diff, used=list(per))


# ── main estimation function ──────────────────────────────────────────────────

def estimate_drift(
    scans: list,
    ref_scan: Any,
    *,
    image_key: str = "adf+bf",
    upsample_factor: int = 20,
    max_shift_px: float = 50.0,
    strain_label: str = "without_roi",
    roi_bounds: list | None = None,
    mode: str = "reference",
    background_sigma: float = 0.0,
    window: bool = True,
    log=None,
) -> list[dict]:
    """Estimate inter-scan drift of *scans* (in the given order).

    Parameters
    ----------
    scans : list of Scan
        The scans to register, in acquisition order (the user's selection).
    ref_scan : Scan
        Reference, shift (0, 0). In ``chained`` mode the chain starts there and
        runs forwards and backwards along *scans*.
    image_key : "adf+bf" | "adf" | "bf" | "exx" | "eyy" | "exy"
    upsample_factor : sub-pixel precision = 1/upsample px.
    roi_bounds : [x0, x1, y0, y1] tracking window, or None for the whole map.
    mode : "reference" | "chained"
    background_sigma : Gaussian (px) of the smooth background removed; 0 = auto.
    window : taper the edges before the FFT.

    Returns
    -------
    list of dict (one per scan, same order): name, dy, dx, magnitude, error,
    ncc_before, ncc_after, adf_bf_diff_px, is_reference, warning, channel, mode.
    """
    def _log(msg):
        if log:
            log(msg)

    scans = list(scans)
    if ref_scan not in scans:
        scans.insert(0, ref_scan)
    key = image_key.lower()
    if key != "adf+bf" and _get_map(ref_scan, key, strain_label) is None:
        raise RuntimeError(
            f"Reference scan '{getattr(ref_scan, 'name', '?')}' has no '{key}' map. "
            "Load the ADF/BF (virtual images) or compute strain first.")
    if key == "adf+bf" and _get_map(ref_scan, "adf", strain_label) is None \
            and _get_map(ref_scan, "bf", strain_label) is None:
        raise RuntimeError(
            f"Reference scan '{getattr(ref_scan, 'name', '?')}' has neither ADF nor BF.")

    roi_note = (f"tracking ROI {[int(b) for b in roi_bounds]}" if roi_bounds else "whole map")
    _log(f"[Drift] reference '{getattr(ref_scan, 'name', '?')}' | channel {key} | "
         f"mode {mode} | upsample {upsample_factor}x | {roi_note}")

    args = (key, strain_label, roi_bounds, upsample_factor, background_sigma, window)
    iref = scans.index(ref_scan)
    res: dict[int, dict] = {iref: dict(dy=0.0, dx=0.0, error=0.0, ncc_before=1.0,
                                       ncc_after=1.0, adf_bf_diff_px=0.0)}
    if mode == "chained":
        # forwards and backwards from the reference, accumulating each link
        for step in (1, -1):
            i = iref
            acc = np.zeros(2)
            while 0 <= i + step < len(scans):
                j = i + step
                p = _pair(scans[i], scans[j], *args)
                if p is None:
                    res[j] = None
                    break
                acc = acc + np.array([p["dy"], p["dx"]])
                # quality against the reference, with the accumulated shift
                q = _pair(ref_scan, scans[j], *args)
                p = dict(p, dy=float(acc[0]), dx=float(acc[1]))
                if q is not None:
                    p["ncc_before"] = q["ncc_before"]
                    ref_map = _get_map(ref_scan, key if key != "adf+bf" else "adf", strain_label)
                    mov_map = _get_map(scans[j], key if key != "adf+bf" else "adf", strain_label)
                    if ref_map is not None and mov_map is not None:
                        rq = condition(_prep_image(ref_map), background_sigma=background_sigma,
                                       window=False)
                        mq = condition(_prep_image(mov_map), background_sigma=background_sigma,
                                       window=False)
                        p["ncc_after"] = ncc(rq, shift_image(mq, acc[0], acc[1]))
                    p["link_ncc_after"] = q["ncc_after"]
                res[j] = p
                i = j
    else:
        for j, sc in enumerate(scans):
            if j == iref:
                continue
            res[j] = _pair(ref_scan, sc, *args)

    out = []
    for j, sc in enumerate(scans):
        name = getattr(sc, "name", str(sc))
        p = res.get(j)
        if p is None:
            _log(f"[Drift] '{name}': no '{key}' map — zero shift assumed.")
            out.append(dict(name=name, dy=0.0, dx=0.0, magnitude=0.0, error=1.0,
                            ncc_before=float("nan"), ncc_after=float("nan"),
                            adf_bf_diff_px=float("nan"), is_reference=False,
                            warning=f"no '{key}' map", channel=key, mode=mode))
            continue
        mag = float(np.hypot(p["dy"], p["dx"]))
        warn = []
        if mag > max_shift_px:
            warn.append(f"large shift ({mag:.1f} px)")
        if j != iref and np.isfinite(p["ncc_after"]) and p["ncc_after"] < 0.5:
            warn.append(f"poor match (NCC {p['ncc_after']:.2f})")
        if np.isfinite(p.get("adf_bf_diff_px", np.nan)) and p["adf_bf_diff_px"] > 1.0:
            warn.append(f"ADF/BF disagree by {p['adf_bf_diff_px']:.1f} px")
        r = dict(name=name, dy=p["dy"], dx=p["dx"], magnitude=mag, error=p["error"],
                 ncc_before=p["ncc_before"], ncc_after=p["ncc_after"],
                 adf_bf_diff_px=p.get("adf_bf_diff_px", float("nan")),
                 is_reference=(j == iref), warning="; ".join(warn), channel=key, mode=mode)
        out.append(r)
        _log(f"[Drift] '{name}': " + ("reference (0.00, 0.00)" if j == iref else
             f"dy={p['dy']:+.2f} dx={p['dx']:+.2f} px  NCC {p['ncc_before']:.2f} -> "
             f"{p['ncc_after']:.2f}" + (f"  ADF/BF diff {p['adf_bf_diff_px']:.2f} px"
                                        if np.isfinite(p.get('adf_bf_diff_px', np.nan)) else "")
             + (f"  ⚠ {r['warning']}" if r["warning"] else "")))
    return out


def summary(results: list[dict], px_nm: float | None = None) -> dict:
    """Numbers for the reproducibility report: shift spread and the mean drift
    step between consecutive scans (in px, and nm when the step is known)."""
    d = np.array([[r["dy"], r["dx"]] for r in results], float)
    steps = np.hypot(*np.diff(d, axis=0).T) if len(d) > 1 else np.array([])
    s = dict(n=len(results),
             max_shift_px=float(np.hypot(d[:, 0], d[:, 1]).max()) if len(d) else 0.0,
             mean_step_px=float(steps.mean()) if steps.size else 0.0,
             ncc_after_min=float(np.nanmin([r["ncc_after"] for r in results
                                            if not r["is_reference"]] or [np.nan])))
    if px_nm:
        s.update(max_shift_nm=s["max_shift_px"] * px_nm, mean_step_nm=s["mean_step_px"] * px_nm)
    return s


# ── overlays (evidence of the drift) ──────────────────────────────────────────

def overlay_colors(n: int) -> list[tuple[float, float, float]]:
    """One colour per scan. 2 scans: magenta / green; 3: red / green / blue.
    In both cases perfectly overlapping features come out grey/white."""
    if n <= 1:
        return [(1.0, 1.0, 1.0)]
    if n == 2:
        return [(1.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
    if n == 3:
        return [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    import colorsys
    return [colorsys.hsv_to_rgb(i / n, 1.0, 1.0) for i in range(n)]


def _norm01(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    fin = np.isfinite(a)
    if not fin.any():
        return np.zeros_like(a)
    lo, hi = np.percentile(a[fin], (1, 99))
    return np.where(fin, np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1), 0.0)


def composite(images: list, shifts: list | None = None,
              colors: list | None = None) -> np.ndarray:
    """RGB overlay of several scans (each in its own colour). With *shifts*
    [(dy, dx), ...] every image is first registered, so the overlay after the
    correction turns grey/white where the scans agree and stays coloured where
    they do not; without shifts it shows the raw drift as coloured fringes."""
    ims = [np.asarray(i, dtype=np.float64) for i in images]
    h = min(i.shape[0] for i in ims)
    w = min(i.shape[1] for i in ims)
    ims = [i[:h, :w] for i in ims]
    if shifts is not None:
        ims = [shift_image(i, dy, dx) for i, (dy, dx) in zip(ims, shifts)]
    cols = np.asarray(colors or overlay_colors(len(ims)), dtype=np.float64)
    rgb = np.zeros((h, w, 3))
    for im, c in zip(ims, cols):
        rgb += _norm01(im)[..., None] * c[None, None, :]
    rgb /= np.maximum(cols.sum(axis=0), 1e-12)[None, None, :]
    if shifts is not None:              # only where every scan has data after the shift
        common = np.logical_and.reduce([np.isfinite(i) for i in ims])
        rgb[~common] = 0.0
    return np.clip(rgb, 0, 1)


def overlay_maps(scans: list, results: list[dict], image_key: str = "adf+bf",
                 label: str = "without_roi") -> tuple[np.ndarray, np.ndarray, str, list]:
    """(before, after, channel_used, names): the two overlays for *results*
    (the output of estimate_drift) — the raw scans and the drift-corrected ones."""
    keys = ("adf", "bf") if image_key == "adf+bf" else (image_key,)
    by_name = {getattr(s, "name", str(s)): s for s in scans}
    for k in keys:
        maps, shifts, names = [], [], []
        for r in results:
            sc = by_name.get(r["name"])
            m = _get_map(sc, k, label) if sc is not None else None
            if m is None:
                maps = []
                break
            maps.append(m); shifts.append((r["dy"], r["dx"])); names.append(r["name"])
        if len(maps) >= 2:
            return composite(maps), composite(maps, shifts), k, names
    raise RuntimeError("no common ADF/BF map to overlay")


def save_overlays(before: np.ndarray, after: np.ndarray, names: list, folder: str | Path,
                  *, stem: str = "drift_overlay", channel: str = "adf",
                  results: list[dict] | None = None, px_nm: float | None = None) -> list[Path]:
    """Write <stem>_before.png and <stem>_after.png with a colour legend."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    cols = overlay_colors(len(names))
    out = []
    shifts = {r["name"]: r for r in (results or [])}
    for tag, img in (("before", before), ("after", after)):
        fig = Figure(figsize=(6.4, 7.0), dpi=200)
        ax = fig.add_axes([0.02, 0.14, 0.96, 0.80])
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        lab = []
        for n, c in zip(names, cols):
            r = shifts.get(n)
            txt = n
            if r is not None and tag == "after" and not r.get("is_reference"):
                txt += f"  ({r['dy']:+.2f}, {r['dx']:+.2f}) px"
                if px_nm:
                    txt += f" = {r['magnitude'] * px_nm:.2f} nm"
            elif r is not None and r.get("is_reference"):
                txt += "  (reference)"
            lab.append(Patch(facecolor=c, edgecolor="k", label=txt))
        fig.legend(handles=lab, loc="lower center", ncol=1 if len(lab) > 3 else len(lab),
                   fontsize=7, frameon=False)
        fig.text(0.02, 0.975, f"{channel.upper()} · {'as acquired (drift visible)' if tag == 'before' else 'after drift correction'}",
                 va="top", fontsize=8)
        p = folder / f"{stem}_{tag}.png"
        fig.savefig(p)
        out.append(p)
    return out


# ── CSV export ────────────────────────────────────────────────────────────────

def save_drift_csv(results: list[dict], path: str | Path, *, reference: str | None = None) -> Path:
    """Write drift results to a CSV compatible with ``engine.load_drift_csv``.

    Columns: stem, shift_dy_px, shift_dx_px, then QC extras the loader ignores.
    (No extra column is called name/file/scan: the loader would take it as the
    key column.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = reference or next((r["name"] for r in results if r.get("is_reference")), "")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "shift_dy_px", "shift_dx_px", "channel", "mode", "reference_stem",
                    "ncc_before", "ncc_after", "adf_bf_diff_px", "warning"])
        for r in results:
            fmt = lambda v: "" if v is None or not np.isfinite(v) else f"{v:.4f}"
            w.writerow([r["name"], f"{r['dy']:.4f}", f"{r['dx']:.4f}", r.get("channel", ""),
                        r.get("mode", ""), ref, fmt(r.get("ncc_before")), fmt(r.get("ncc_after")),
                        fmt(r.get("adf_bf_diff_px")), r.get("warning", "")])
    return path
