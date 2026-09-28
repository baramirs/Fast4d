"""fast4d.reproducibility — "Reproducibility experiment" report.

The same region is scanned N times in a row, nothing changed on the microscope,
to measure how stable the measurement is. The sample drifts between scans, so
every metric is computed twice:

* **no drift**   — the same pixel coordinates in every scan (what you get if
  you ignore the drift);
* **with drift** — every scan brought onto the reference scan with the
  per-scan drift (``scan.drift = (dx, dy)``, the registration shift written by
  the drift plugin), so the same PHYSICAL position is compared.

The difference between the two columns is the part of the apparent
irreproducibility that is only drift.

Metrics (per strain channel εxx/εyy/εxy; ADF/BF only where it makes sense):

Lines (the reference scan's line, e.g. L1, placed on every scan)
  profile of every scan · mean ± std across scans at every point · mean of each
  scan's line · across-scan std / SE / CV of those means · mean pointwise
  across-scan std s(d) · RMSE and offset of each scan vs the reference.
ROIs (the reference scan's area ROIs)
  mean per scan · across-scan std / CV · contrast between two ROIs (A − B) per
  scan and its spread.
Maps
  offset = mean(ε_j − ε_ref) and RMSE = sqrt(mean((ε_j − ε_ref)²)) of every scan
  vs the reference; repeatability map s(r) = std over scans at each registered
  position (ddof = 1) with its median / mean over the common valid area.
Fit quality
  fraction of scan positions with a valid fit (Fast4D writes 0 in all three
  channels where the fit failed; those positions are excluded everywhere).
Drift
  (dx, dy) per scan in px and nm, step between consecutive scans and, when the
  raw files carry distinct modification times, a drift velocity (nm/min).

Note: pixels are not independent repetitions — the across-scan numbers use N
scans, not the number of pixels.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

STRAIN = ("exx", "eyy", "exy")
_IDX = {"eyy": 0, "exx": 1, "exy": 2}
LABEL = {"exx": "ε_xx (%)", "eyy": "ε_yy (%)", "exy": "ε_xy (%)", "adf": "ADF", "bf": "BF"}
MODES = ("no_drift", "drift")
MODE_TXT = {"no_drift": "no drift correction", "drift": "drift corrected"}
COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
          "#f032e6", "#9A6324", "#808000", "#000075"]


# ── maps ──────────────────────────────────────────────────────────────────────

def _hw3(scan, label):
    st = getattr(scan, "state", None)
    raw = (getattr(st, "strain_raw", {}) or {}).get(label) if st is not None else None
    if raw is None:
        return None
    try:
        from fast_artifacts import _as_hw3
        return _as_hw3(raw)
    except Exception:
        a = np.asarray(raw, float)
        return a if a.ndim == 3 and a.shape[2] >= 3 else None


def valid_mask(scan, label="without_roi"):
    """True where the strain fit exists (not the all-zero fill, finite)."""
    h = _hw3(scan, label)
    if h is None:
        return None
    h = np.asarray(h[..., :3], float)
    return np.isfinite(h).all(-1) & ~(h == 0).all(-1)


def channel_map(scan, channel, label="without_roi"):
    """Strain in % with invalid positions as NaN; ADF / BF as stored."""
    if channel in _IDX:
        h = _hw3(scan, label)
        if h is None:
            return None
        a = np.asarray(h[..., _IDX[channel]], float) * 100.0
        m = valid_mask(scan, label)
        return np.where(m, a, np.nan)
    import drift_estimate as DE
    a = DE._get_map(scan, channel, label)
    return None if a is None else np.asarray(a, float)


def _drift(scan):
    d = getattr(scan, "drift", None)
    return (float(d[0]), float(d[1])) if d else (0.0, 0.0)


def rel_shift(scan, ref):
    """Registration shift (dy, dx) that brings *scan* onto *ref*'s pixel frame."""
    dxs, dys = _drift(scan)
    dxr, dyr = _drift(ref)
    return dys - dyr, dxs - dxr


def register(arr, dy, dx):
    """Shift a map by the registration shift (NaN outside and NaN-aware)."""
    from scipy.ndimage import shift as _sh
    a = np.asarray(arr, float)
    if abs(dy) < 1e-9 and abs(dx) < 1e-9:
        return a.copy()
    ok = np.isfinite(a)
    v = _sh(np.where(ok, a, 0.0), (dy, dx), order=1, mode="constant", cval=0.0)
    w = _sh(ok.astype(float), (dy, dx), order=1, mode="constant", cval=0.0)
    return np.where(w > 0.99, v / np.maximum(w, 1e-12), np.nan)


def _crop_common(arrs):
    h = min(a.shape[0] for a in arrs)
    w = min(a.shape[1] for a in arrs)
    return [a[:h, :w] for a in arrs]


# ── line sampling (NaN-aware, same geometry as engine._line_samples) ──────────

def line_samples(arr2d, seg, width=3):
    from scipy.ndimage import map_coordinates
    (x0, y0), (x1, y1) = seg
    L = float(np.hypot(x1 - x0, y1 - y0))
    n = int(max(2, round(L) + 1))
    t = np.linspace(0.0, 1.0, n)
    xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    px, py = ((-(y1 - y0) / L, (x1 - x0) / L) if L > 0 else (0.0, 0.0))
    a = np.asarray(arr2d, float)
    ok = np.isfinite(a)
    fa = np.where(ok, a, 0.0)
    acc = np.zeros(n); wacc = np.zeros(n)
    w = max(1, int(width))
    for o in np.arange(w) - (w - 1) / 2.0:
        cy, cx = ys + o * py, xs + o * px
        inside = (cy >= 0) & (cy <= a.shape[0] - 1) & (cx >= 0) & (cx <= a.shape[1] - 1)
        v = map_coordinates(fa, [cy, cx], order=1, mode="nearest")
        m = map_coordinates(ok.astype(float), [cy, cx], order=1, mode="nearest")
        acc += np.where(inside, v, 0.0); wacc += np.where(inside, m, 0.0)
    vals = np.where(wacc > 0.5 * w, acc / np.maximum(wacc, 1e-12), np.nan)
    return t * L, vals


def shift_segment(seg, dx, dy):
    return [[seg[0][0] + dx, seg[0][1] + dy], [seg[1][0] + dx, seg[1][1] + dy]]


def _roi_mean(arr, bounds, dx=0.0, dy=0.0):
    x0, x1, y0, y1 = [float(b) for b in bounds]
    H, W = arr.shape
    xa, xb = sorted((x0 + dx, x1 + dx)); ya, yb = sorted((y0 + dy, y1 + dy))
    xa, ya = max(0, int(round(xa))), max(0, int(round(ya)))
    xb, yb = min(W, int(round(xb))), min(H, int(round(yb)))
    if xb <= xa or yb <= ya:
        return float("nan"), 0
    v = arr[ya:yb, xa:xb]
    v = v[np.isfinite(v)]
    return (float(v.mean()) if v.size else float("nan")), int(v.size)


def _stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    n = v.size
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, sem=np.nan, cv_pct=np.nan)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    return dict(n=n, mean=mean, std=std, sem=std / math.sqrt(n) if n else np.nan,
                cv_pct=(std / abs(mean) * 100.0) if abs(mean) > 1e-12 else np.nan)


# ── the experiment ────────────────────────────────────────────────────────────

@dataclass
class Result:
    names: list
    ref: str
    label: str
    px_nm: float | None
    channels: list
    drift: list = field(default_factory=list)          # per-scan rows
    quality: list = field(default_factory=list)
    lines: dict = field(default_factory=dict)          # lid -> ch -> mode -> dict
    rois: dict = field(default_factory=dict)           # rid -> ch -> mode -> dict
    contrast: dict = field(default_factory=dict)       # "R1-R2" -> ch -> mode -> dict
    maps: dict = field(default_factory=dict)           # ch -> mode -> dict
    has_drift: bool = False
    notes: list = field(default_factory=list)


def _px_nm(scan):
    p = getattr(scan, "params", None)
    pm = float(getattr(p, "r_px_pm", 0.0) or 0.0) if p is not None else 0.0
    return pm / 1000.0 if pm > 0 else None


def _file_time(scan):
    for a in ("raw_path", "h5_path", "braggpeaks_path"):
        p = getattr(scan, a, None)
        if p and os.path.isfile(str(p)):
            try:
                return os.path.getmtime(str(p))
            except OSError:
                pass
    return None


def run(scans: list, ref, **kw) -> Result:
    """Compute every metric for *scans* (acquisition order) against *ref*."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns are expected
        return _run(scans, ref, **kw)


def _run(scans: list, ref, *, label="without_roi", width=3, channels=None,
         line_ids=None, roi_ids=None, images=("adf", "bf"), log=None) -> Result:
    def _log(m):
        if log:
            log(m)
    scans = list(scans)
    if ref not in scans:
        scans.insert(0, ref)
    channels = [c for c in (channels or STRAIN) if c in STRAIN]
    names = [getattr(s, "name", str(s)) for s in scans]
    px = _px_nm(ref)
    res = Result(names=names, ref=getattr(ref, "name", "?"), label=label, px_nm=px,
                 channels=channels)
    res.has_drift = any(getattr(s, "drift", None) for s in scans)
    if not res.has_drift:
        res.notes.append("No drift loaded: 'drift corrected' equals 'no drift correction'. "
                         "Estimate drift first (Tools → Inter-scan drift).")

    # drift table ---------------------------------------------------------------
    times = [_file_time(s) for s in scans]
    prev = None
    for i, s in enumerate(scans):
        dy, dx = rel_shift(s, ref)
        mag = math.hypot(dx, dy)
        step = math.hypot(dx - prev[1], dy - prev[0]) if prev is not None else float("nan")
        row = dict(scan=names[i], dx_px=dx, dy_px=dy, shift_px=mag, step_px=step)
        if px:
            row.update(dx_nm=dx * px, dy_nm=dy * px, shift_nm=mag * px,
                       step_nm=step * px if np.isfinite(step) else float("nan"))
        if i > 0 and times[i] and times[i - 1] and times[i] > times[i - 1] and px \
                and np.isfinite(step):
            row["velocity_nm_per_min"] = step * px / ((times[i] - times[i - 1]) / 60.0)
        res.drift.append(row)
        prev = (dy, dx)
    if any("velocity_nm_per_min" in r for r in res.drift):
        res.notes.append("Drift velocity uses the raw files' modification times "
                         "(end of each acquisition) — check against the logbook.")

    # fit quality -----------------------------------------------------------------
    for s, n in zip(scans, names):
        m = valid_mask(s, label)
        res.quality.append(dict(scan=n, valid_fraction=float(m.mean()) if m is not None else np.nan,
                                n_valid=int(m.sum()) if m is not None else 0))

    # maps: offset / RMSE vs reference, repeatability map --------------------------
    for ch in channels + [k for k in images]:
        raw = [channel_map(s, ch, label) for s in scans]
        if any(a is None for a in raw):
            continue
        raw = _crop_common(raw)
        res.maps[ch] = {}
        for mode in MODES:
            arrs = [register(a, *rel_shift(s, ref)) if mode == "drift" else a
                    for a, s in zip(raw, scans)]
            stack = np.stack(arrs)
            common = np.isfinite(stack).all(0)
            iref = scans.index(ref)
            per = []
            for i, a in enumerate(arrs):
                d = (a - arrs[iref])[common]
                per.append(dict(scan=names[i], offset=float(d.mean()) if d.size else np.nan,
                                rmse=float(np.sqrt((d ** 2).mean())) if d.size else np.nan,
                                corr=float(np.corrcoef(a[common], arrs[iref][common])[0, 1])
                                if d.size > 2 and i != iref else (1.0 if i == iref else np.nan),
                                n=int(d.size)))
            s_map = np.where(common, np.nanstd(stack, axis=0, ddof=1), np.nan) \
                if len(arrs) > 1 else np.full(common.shape, np.nan)
            sv = s_map[common]
            res.maps[ch][mode] = dict(per_scan=per, s_map=s_map, common=common,
                                      s_median=float(np.median(sv)) if sv.size else np.nan,
                                      s_mean=float(sv.mean()) if sv.size else np.nan,
                                      n_common=int(common.sum()))
    # lines -----------------------------------------------------------------------
    ref_lines = dict(getattr(ref, "lines", None) or {})
    for lid in (line_ids or sorted(ref_lines)):
        seg = ref_lines.get(lid)
        if seg is None:
            continue
        res.lines[lid] = {}
        for ch in channels + list(images):
            per_mode = {}
            for mode in MODES:
                curves, dist = [], None
                for s in scans:
                    a = channel_map(s, ch, label)
                    if a is None:
                        curves = None
                        break
                    dy, dx = rel_shift(s, ref)
                    sg = shift_segment(seg, -dx, -dy) if mode == "drift" else seg
                    d, v = line_samples(a, sg, width=width)
                    dist = d if dist is None else dist
                    curves.append(v[:len(dist)] if len(v) >= len(dist) else
                                  np.pad(v, (0, len(dist) - len(v)), constant_values=np.nan))
                if not curves:
                    break
                C = np.vstack(curves)
                iref = scans.index(ref)
                with np.errstate(all="ignore"):
                    mean_c = np.nanmean(C, axis=0)
                    std_c = np.nanstd(C, axis=0, ddof=1) if len(curves) > 1 else np.zeros(C.shape[1])
                per = []
                for i, c in enumerate(C):
                    ok = np.isfinite(c) & np.isfinite(C[iref])
                    dd = c[ok] - C[iref][ok]
                    okm = np.isfinite(c) & np.isfinite(mean_c)
                    per.append(dict(scan=names[i], line_mean=float(np.nanmean(c)) if np.isfinite(c).any() else np.nan,
                                    offset_vs_ref=float(dd.mean()) if dd.size else np.nan,
                                    rmse_vs_ref=float(np.sqrt((dd ** 2).mean())) if dd.size else np.nan,
                                    rmse_vs_mean=float(np.sqrt(((c[okm] - mean_c[okm]) ** 2).mean()))
                                    if okm.any() else np.nan))
                st = _stats([p["line_mean"] for p in per])
                per_mode[mode] = dict(dist=dist, curves=C, mean=mean_c, std=std_c, per_scan=per,
                                      summary=st,
                                      s_mean=float(np.nanmean(std_c)) if np.isfinite(std_c).any() else np.nan)
            if per_mode:
                res.lines[lid][ch] = per_mode
    # ROIs and contrast -----------------------------------------------------------
    try:
        import engine as _E
        ref_rois = _E.scan_area_rois(ref)
    except Exception:
        ref_rois = dict(getattr(ref, "area_rois", None) or {})
    ref_rois = {k: v for k, v in ref_rois.items() if (roi_ids is None or k in roi_ids)}
    for rid, b in ref_rois.items():
        res.rois[rid] = {}
        for ch in channels:
            res.rois[rid][ch] = {}
            for mode in MODES:
                vals = []
                for s in scans:
                    a = channel_map(s, ch, label)
                    dy, dx = rel_shift(s, ref)
                    m, n = _roi_mean(a, b, *((-dx, -dy) if mode == "drift" else (0.0, 0.0))) \
                        if a is not None else (np.nan, 0)
                    vals.append(m)
                res.rois[rid][ch][mode] = dict(values=vals, summary=_stats(vals))
    rids = list(res.rois)
    for i in range(len(rids)):
        for j in range(i + 1, len(rids)):
            key = f"{rids[i]}-{rids[j]}"
            res.contrast[key] = {}
            for ch in channels:
                res.contrast[key][ch] = {}
                for mode in MODES:
                    a = res.rois[rids[i]][ch][mode]["values"]
                    b = res.rois[rids[j]][ch][mode]["values"]
                    v = [x - y for x, y in zip(a, b)]
                    res.contrast[key][ch][mode] = dict(values=v, summary=_stats(v))
    _log(f"[Repro] {len(scans)} scans, ref {res.ref}: {len(res.lines)} line(s), "
         f"{len(res.rois)} ROI(s), maps {list(res.maps)}")
    return res


# ── tables ────────────────────────────────────────────────────────────────────

def summary_table(res: Result):
    """Long table: object, channel, metric, no_drift, drift, unit."""
    import pandas as pd
    rows = []

    def add(obj, ch, metric, nd, dr, unit):
        rows.append(dict(object=obj, channel=ch, metric=metric, no_drift=nd, drift=dr, unit=unit))
    u = lambda ch: "%" if ch in STRAIN else "counts"
    for lid, chs in res.lines.items():
        for ch, m in chs.items():
            nd, dr = m.get("no_drift"), m.get("drift")
            g = lambda d, k: d["summary"].get(k) if d else np.nan
            add(lid, ch, "mean of line means", g(nd, "mean"), g(dr, "mean"), u(ch))
            add(lid, ch, "std across scans (of line means)", g(nd, "std"), g(dr, "std"), u(ch))
            add(lid, ch, "CV across scans", g(nd, "cv_pct"), g(dr, "cv_pct"), "%")
            add(lid, ch, "mean pointwise std s(d)", nd["s_mean"] if nd else np.nan,
                dr["s_mean"] if dr else np.nan, u(ch))
            rm = lambda d: float(np.nanmean([p["rmse_vs_ref"] for p in d["per_scan"]
                                              if p["scan"] != res.ref])) if d else np.nan
            add(lid, ch, "mean RMSE vs reference", rm(nd), rm(dr), u(ch))
    for rid, chs in res.rois.items():
        for ch, m in chs.items():
            add(rid, ch, "mean over scans", m["no_drift"]["summary"]["mean"],
                m["drift"]["summary"]["mean"], u(ch))
            add(rid, ch, "std across scans", m["no_drift"]["summary"]["std"],
                m["drift"]["summary"]["std"], u(ch))
            add(rid, ch, "CV across scans", m["no_drift"]["summary"]["cv_pct"],
                m["drift"]["summary"]["cv_pct"], "%")
    for key, chs in res.contrast.items():
        for ch, m in chs.items():
            add(f"contrast {key}", ch, "mean (A - B)", m["no_drift"]["summary"]["mean"],
                m["drift"]["summary"]["mean"], u(ch))
            add(f"contrast {key}", ch, "std across scans", m["no_drift"]["summary"]["std"],
                m["drift"]["summary"]["std"], u(ch))
    for ch, m in res.maps.items():
        add("map", ch, "median repeatability s(r)", m["no_drift"]["s_median"],
            m["drift"]["s_median"], u(ch))
        add("map", ch, "mean repeatability s(r)", m["no_drift"]["s_mean"], m["drift"]["s_mean"], u(ch))
        for k in ("offset", "rmse"):
            vals = lambda mode: [p[k] for p in m[mode]["per_scan"] if p["scan"] != res.ref]
            add("map", ch, f"mean |{k}| vs reference" if k == "offset" else "mean RMSE vs reference",
                float(np.nanmean(np.abs(vals("no_drift")))) if vals("no_drift") else np.nan,
                float(np.nanmean(np.abs(vals("drift")))) if vals("drift") else np.nan, u(ch))
        add("map", ch, "common valid positions", m["no_drift"]["n_common"], m["drift"]["n_common"], "px")
    return pd.DataFrame(rows)


def drift_table(res: Result):
    import pandas as pd
    return pd.DataFrame(res.drift)


def quality_table(res: Result):
    import pandas as pd
    return pd.DataFrame(res.quality)


def per_scan_table(res: Result):
    """Every per-scan number: line means / offsets / RMSE, ROI means, map offset / RMSE."""
    import pandas as pd
    rows = []
    for lid, chs in res.lines.items():
        for ch, m in chs.items():
            for mode, d in m.items():
                for p in d["per_scan"]:
                    rows.append(dict(object=lid, channel=ch, mode=mode, **p))
    for rid, chs in res.rois.items():
        for ch, m in chs.items():
            for mode, d in m.items():
                for n, v in zip(res.names, d["values"]):
                    rows.append(dict(object=rid, channel=ch, mode=mode, scan=n, roi_mean=v))
    for ch, m in res.maps.items():
        for mode, d in m.items():
            for p in d["per_scan"]:
                rows.append(dict(object="map", channel=ch, mode=mode, **p))
    return pd.DataFrame(rows)


def line_profiles_table(res: Result, lid: str):
    """Long table of every profile of one line (for your own plots)."""
    import pandas as pd
    rows = []
    for ch, m in res.lines.get(lid, {}).items():
        for mode, d in m.items():
            dist = d["dist"] * (res.px_nm or 1.0)
            for k, name in enumerate(res.names):
                for x, v in zip(dist, d["curves"][k]):
                    rows.append(dict(line=lid, channel=ch, mode=mode, scan=name, distance=x, value=v))
            for x, v, s in zip(dist, d["mean"], d["std"]):
                rows.append(dict(line=lid, channel=ch, mode=mode, scan="MEAN", distance=x, value=v, std=s))
    return pd.DataFrame(rows)


# ── figures ───────────────────────────────────────────────────────────────────

def line_figure(res: Result, lid: str):
    """Rows = channels (strain, then ADF/BF), columns = no drift | drift corrected.
    Every scan's profile, the mean and the ±1 std band across scans."""
    from matplotlib.figure import Figure
    chs = list(res.lines.get(lid, {}))
    n = max(1, len(chs))
    fig = Figure(figsize=(11, 2.5 * n + 0.8), constrained_layout=True)
    if not chs:
        ax = fig.add_subplot(111); ax.text(0.5, 0.5, f"no data for {lid}", ha="center"); ax.axis("off")
        return fig
    axs = fig.subplots(n, 2, squeeze=False, sharex=True)
    unit = "nm" if res.px_nm else "px"
    for r, ch in enumerate(chs):
        lo, hi = [], []
        for c, mode in enumerate(MODES):
            ax = axs[r][c]
            d = res.lines[lid][ch][mode]
            x = d["dist"] * (res.px_nm or 1.0)
            for k, name in enumerate(res.names):
                ax.plot(x, d["curves"][k], color=COLORS[k % len(COLORS)], lw=1.0, alpha=0.85,
                        label=name + (" (ref)" if name == res.ref else ""))
            ax.fill_between(x, d["mean"] - d["std"], d["mean"] + d["std"], color="0.5", alpha=0.25,
                            lw=0, label="mean ± std")
            ax.plot(x, d["mean"], color="k", lw=1.8, label="mean")
            fin = np.isfinite(d["curves"])
            if fin.any():
                lo.append(np.nanpercentile(d["curves"], 1)); hi.append(np.nanpercentile(d["curves"], 99))
            s = d["summary"]
            if ch in STRAIN:
                txt = (f"line mean {s['mean']:.4f} ± {s['std']:.4f} %  (CV {s['cv_pct']:.2g} %)\n"
                       f"mean pointwise std {d['s_mean']:.4f} %")
            else:
                txt = f"mean pointwise std {d['s_mean']:.4g}"
            ax.text(0.01, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
            if r == 0:
                ax.set_title(f"{lid} — {MODE_TXT[mode]}", fontsize=10)
            if c == 0:
                ax.set_ylabel(LABEL.get(ch, ch))
            if r == n - 1:
                ax.set_xlabel(f"distance ({unit})")
            ax.grid(alpha=0.3)
        if lo:
            pad = 0.05 * (max(hi) - min(lo) or 1)
            for c in range(2):
                axs[r][c].set_ylim(min(lo) - pad, max(hi) + pad)
    axs[0][1].legend(fontsize=7, loc="lower right", ncol=2)
    return fig


def map_figure(res: Result, ch: str):
    """Repeatability map s(r) without and with drift correction (same colour
    scale) and the distribution of s over the common area."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(11, 4.2), constrained_layout=True)
    m = res.maps.get(ch)
    if not m:
        ax = fig.add_subplot(111); ax.text(0.5, 0.5, f"no {ch} maps", ha="center"); ax.axis("off")
        return fig
    axs = fig.subplots(1, 3)
    allv = np.concatenate([m[k]["s_map"][np.isfinite(m[k]["s_map"])] for k in MODES])
    vmax = float(np.percentile(allv, 99)) if allv.size else 1.0
    for ax, mode in zip(axs[:2], MODES):
        im = ax.imshow(m[mode]["s_map"], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"s(r) {LABEL.get(ch, ch)} — {MODE_TXT[mode]}\n"
                     f"median {m[mode]['s_median']:.4g} · n {m[mode]['n_common']}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs[:2], shrink=0.8, label=f"std across scans ({'%' if ch in STRAIN else 'a.u.'})")
    ax = axs[2]
    bins = np.linspace(0, vmax, 50)
    for mode, col in zip(MODES, ("#C62828", "#1565C0")):
        v = m[mode]["s_map"][np.isfinite(m[mode]["s_map"])]
        ax.hist(v, bins=bins, histtype="step", lw=1.6, color=col, density=True, label=MODE_TXT[mode])
    ax.set_xlabel("s(r)"); ax.set_ylabel("density"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("distribution over the common area", fontsize=9)
    return fig


def drift_figure(res: Result):
    """Drift trajectory of the scans relative to the reference (feature motion)."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5.2, 4.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    s = res.px_nm or 1.0
    unit = "nm" if res.px_nm else "px"
    # feature displacement = minus the registration shift
    x = [-r["dx_px"] * s for r in res.drift]; y = [-r["dy_px"] * s for r in res.drift]
    ax.plot(x, y, "-", color="0.6", lw=1)
    for k, (xi, yi) in enumerate(zip(x, y)):
        ax.plot(xi, yi, "o", color=COLORS[k % len(COLORS)], ms=8)
        ax.annotate(res.names[k], (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel(f"feature displacement x ({unit})"); ax.set_ylabel(f"y ({unit}, rows down)")
    ax.invert_yaxis(); ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
    ax.set_title("sample drift between scans (vs reference)", fontsize=9)
    return fig


def export(res: Result, folder: str, *, stem="reproducibility") -> list:
    """Write every table (CSV) and figure (PNG) into *folder*."""
    from pathlib import Path
    out = []
    d = Path(folder); d.mkdir(parents=True, exist_ok=True)
    for name, df in (("summary", summary_table(res)), ("per_scan", per_scan_table(res)),
                     ("drift", drift_table(res)), ("fit_quality", quality_table(res))):
        p = d / f"{stem}_{name}.csv"; df.to_csv(p, index=False, encoding="utf-8-sig"); out.append(p)
    for lid in res.lines:
        p = d / f"{stem}_line_{lid}_profiles.csv"
        line_profiles_table(res, lid).to_csv(p, index=False, encoding="utf-8-sig"); out.append(p)
        p = d / f"{stem}_line_{lid}.png"; line_figure(res, lid).savefig(p, dpi=200); out.append(p)
    for ch in res.maps:
        p = d / f"{stem}_map_{ch}.png"; map_figure(res, ch).savefig(p, dpi=200); out.append(p)
    p = d / f"{stem}_drift.png"; drift_figure(res).savefig(p, dpi=200); out.append(p)
    if res.notes:
        p = d / f"{stem}_notes.txt"; p.write_text("\n".join(res.notes), encoding="utf-8"); out.append(p)
    return out
