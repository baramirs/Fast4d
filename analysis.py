"""fast4d.analysis — scientific / statistical analysis layer (LIGHT, post-compute).

Operates on the saved strain tensors (``scan.state.strain_raw[label]``, shape
(H,W,3), channels 0=εyy 1=εxx 2=εxy) — never recomputes maps. Built on the
scientific stack available in the env:

    numpy · scipy.stats · pandas · scikit-learn · uncertainties · matplotlib

Every figure-producing function returns a matplotlib ``Figure`` so it can be
registered into the Report via ``engine.register_figure(scan, key, fig)``.

Capabilities (user-selected):
  1. Map statistics        — per-channel mean/std/median/percentiles/skew/kurtosis,
                             single map and cross-scan table.
  2. Distributions         — histogram + KDE; box/violin per channel across scans.
  3. Error propagation     — strain→stress with `uncertainties` on the constants.
  4. Correlation / PCA     — channel correlation; PCA over per-scan feature vectors.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# channel index → label (matches notebook cell 43)
CH_INDEX = {"eyy": 0, "exx": 1, "exy": 2}
CH_LABEL = {"eyy": "ε_yy", "exx": "ε_xx", "exy": "ε_xy"}
CHANNELS = ("eyy", "exx", "exy")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _as_hw3(arr: Any) -> "np.ndarray | None":
    try:
        from fast_artifacts import _as_hw3 as _coerce
    except ImportError:
        from .fast_artifacts import _as_hw3 as _coerce
    return _coerce(arr)


def _strain_of(scan: Any, label: str) -> "np.ndarray | None":
    sr = getattr(getattr(scan, "state", None), "strain_raw", {}) or {}
    return _as_hw3(sr.get(label))


def _channel_values(scan: Any, label: str, ch: str,
                    roi_mask: "np.ndarray | None" = None,
                    *, as_percent: bool = True) -> "np.ndarray | None":
    """Flat finite values of one channel over a map (optionally within ROI)."""
    hw3 = _strain_of(scan, label)
    if hw3 is None:
        return None
    data = np.asarray(hw3[..., CH_INDEX[ch]], dtype=float)
    if roi_mask is not None:
        m = np.asarray(roi_mask, dtype=bool)
        if m.shape == data.shape:
            data = data[m]
    data = data[np.isfinite(data)]
    if as_percent:
        data = data * 100.0
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 1. Map statistics
# ─────────────────────────────────────────────────────────────────────────────

def map_statistics(scan: Any, label: str = "without_roi",
                   roi_mask: "np.ndarray | None" = None,
                   *, as_percent: bool = True) -> dict:
    """Per-channel descriptive stats for one scan's strain map.

    Returns {channel: {mean, std, median, p5, p95, min, max, skew, kurtosis, n}}.
    Values in % when as_percent (matches the GUI's strain display).
    """
    from scipy import stats as _st
    out: dict = {}
    for ch in CHANNELS:
        v = _channel_values(scan, label, ch, roi_mask, as_percent=as_percent)
        if v is None or v.size == 0:
            continue
        out[ch] = {
            "mean":     float(np.mean(v)),
            "std":      float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "median":   float(np.median(v)),
            "p5":       float(np.percentile(v, 5)),
            "p95":      float(np.percentile(v, 95)),
            "min":      float(np.min(v)),
            "max":      float(np.max(v)),
            "skew":     float(_st.skew(v)) if v.size > 2 else 0.0,
            "kurtosis": float(_st.kurtosis(v)) if v.size > 3 else 0.0,
            "n":        int(v.size),
        }
    return out


def cross_scan_stats(scans: list, label: str = "without_roi",
                     roi_mask: "np.ndarray | None" = None,
                     *, as_percent: bool = True):
    """Cross-scan table — one row per (scan, channel) with mean/std/SEM/CI95.

    Returns a pandas.DataFrame. CI95 uses Student-t (scipy.stats.t).
    """
    import pandas as pd
    from scipy import stats as _st

    rows: list[dict] = []
    for sc in scans:
        for ch in CHANNELS:
            v = _channel_values(sc, label, ch, roi_mask, as_percent=as_percent)
            if v is None or v.size == 0:
                continue
            n = v.size
            mean = float(np.mean(v))
            std = float(np.std(v, ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 0 else 0.0
            if n > 1:
                tcrit = float(_st.t.ppf(0.975, n - 1))
                ci = tcrit * sem
            else:
                ci = 0.0
            cv = (std / abs(mean) * 100.0) if abs(mean) > 1e-12 else float("nan")
            rows.append({
                "scan": getattr(sc, "name", "?"), "channel": ch,
                "mean": round(mean, 6), "std": round(std, 6),
                "sem": round(sem, 6),                       # SE of the mean
                "cv_%": round(cv, 3),                       # coefficient of variation (%)
                "ci95": round(ci, 6),                       # half-width
                "ci95_lo": round(mean - ci, 6),             # lower 95% CI of mean
                "ci95_hi": round(mean + ci, 6),             # upper 95% CI of mean
                "median": round(float(np.median(v)), 6), "n": n,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Distributions  (histogram + KDE, box/violin)
# ─────────────────────────────────────────────────────────────────────────────

def distribution_figure(scans: list, channel: str = "eyy",
                        label: str = "without_roi", *, bins: int = 80,
                        roi_mask: "np.ndarray | None" = None):
    """Histogram + Gaussian-KDE of one strain channel, one curve per scan."""
    from matplotlib.figure import Figure
    from scipy import stats as _st

    fig = Figure(figsize=(7, 4.2), constrained_layout=True)
    ax = fig.add_subplot(111)
    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00",
              "#8E24AA", "#00ACC1", "#F4511E", "#3949AB"]
    # x-axis range = the strain calibration display range (params.vrange), so the
    # distribution shares the same scale as the strain maps. None → auto (legacy).
    vr = None
    for sc in scans:
        p = getattr(sc, "params", None)
        if p is not None and getattr(p, "vrange", None):
            vr = [float(p.vrange[0]), float(p.vrange[1])]
            break
    drawn = 0
    for i, sc in enumerate(scans):
        v = _channel_values(sc, label, channel, roi_mask)
        if v is None or v.size < 2:
            continue
        c = colors[i % len(colors)]
        ax.hist(v, bins=bins, range=vr, density=True, alpha=0.25, color=c)
        try:
            kde = _st.gaussian_kde(v)
            lo, hi = vr if vr else (float(np.min(v)), float(np.max(v)))
            xs = np.linspace(lo, hi, 256)
            ax.plot(xs, kde(xs), color=c, lw=1.8,
                    label=getattr(sc, "name", f"scan{i+1}"))
        except Exception:
            pass
        drawn += 1
    if vr:
        ax.set_xlim(vr)
    ax.set_xlabel(f"{CH_LABEL.get(channel, channel)} (%)")
    ax.set_ylabel("density")
    ax.set_title(f"Distribution — {CH_LABEL.get(channel, channel)}  "
                 f"({label}, n={drawn} scan{'s' if drawn != 1 else ''})", fontsize=10)
    if drawn:
        ax.legend(fontsize=7)
    return fig


def boxplot_figure(scans: list, label: str = "without_roi",
                   *, kind: str = "violin",
                   roi_mask: "np.ndarray | None" = None):
    """Box or violin plot per channel across scans (one group per channel)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 4.2), constrained_layout=True)
    axes = fig.subplots(1, 3)
    for ax, ch in zip(axes, CHANNELS):
        data, names = [], []
        for sc in scans:
            v = _channel_values(sc, label, ch, roi_mask)
            if v is not None and v.size:
                data.append(v)
                names.append(getattr(sc, "name", "?")[:8])
        if not data:
            ax.set_axis_off()
            continue
        if kind == "violin":
            ax.violinplot(data, showmeans=True, showextrema=True)
        else:
            ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=45, fontsize=6, ha="right")
        ax.set_title(CH_LABEL.get(ch, ch), fontsize=9)
        ax.axhline(0.0, color="#999", lw=0.6, ls="--")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("strain (%)")
    fig.suptitle(f"{kind.capitalize()} — strain by channel across scans ({label})",
                 fontsize=10)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Error propagation — strain → stress with uncertainties on the constants
# ─────────────────────────────────────────────────────────────────────────────

def strain_to_stress_propagated(scan: Any, label: str = "without_roi",
                                *, c11_gpa: float, c12_gpa: float, c44_gpa: float,
                                c11_err: float = 0.0, c12_err: float = 0.0,
                                c44_err: float = 0.0, mode: str = "plane_stress",
                                roi_mask: "np.ndarray | None" = None) -> dict:
    """Mean stress (GPa) ± propagated error from mean strain and elastic constants.

    Uses the `uncertainties` library on summary (mean) strains and the cubic
    constants. Cubic Hooke; plane_stress reduces σ_zz=0. Returns
    {sigma_xx, sigma_yy, sigma_xy} each as {value, std} in GPa.

    NOTE: propagation is on the *mean* strain per channel (with its SEM as the
    strain uncertainty) — the scientifically meaningful summary, not per-pixel.
    """
    from uncertainties import ufloat

    # mean strain (fraction, not %) + SEM as uncertainty, per channel
    def _mean_sem(ch):
        v = _channel_values(scan, label, ch, roi_mask, as_percent=False)
        if v is None or v.size == 0:
            return ufloat(0.0, 0.0)
        n = v.size
        sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
        return ufloat(float(np.mean(v)), sd / np.sqrt(n) if n else 0.0)

    eyy, exx, exy = _mean_sem("eyy"), _mean_sem("exx"), _mean_sem("exy")
    C11 = ufloat(c11_gpa, c11_err)
    C12 = ufloat(c12_gpa, c12_err)
    C44 = ufloat(c44_gpa, c44_err)

    if mode == "plane_stress":
        # σ_zz = 0 → ε_zz = -C12/(C11)·(εxx+εyy); substitute into σxx, σyy
        ezz = -(C12 / C11) * (exx + eyy)
        sxx = C11 * exx + C12 * (eyy + ezz)
        syy = C11 * eyy + C12 * (exx + ezz)
    else:  # plane_strain → ε_zz = 0
        sxx = C11 * exx + C12 * eyy
        syy = C11 * eyy + C12 * exx
    sxy = 2.0 * C44 * exy

    return {
        "sigma_xx": {"value": sxx.nominal_value, "std": sxx.std_dev},
        "sigma_yy": {"value": syy.nominal_value, "std": syy.std_dev},
        "sigma_xy": {"value": sxy.nominal_value, "std": sxy.std_dev},
        "mode": mode,
    }


def stress_summary_table(scans: list, *, c11_gpa: float, c12_gpa: float,
                         c44_gpa: float, c11_err: float = 0.0, c12_err: float = 0.0,
                         c44_err: float = 0.0, label: str = "without_roi",
                         mode: str = "plane_stress", roi_mask=None):
    """Per-scan propagated stress (GPa) as a pandas.DataFrame (value ± std)."""
    import pandas as pd
    rows = []
    for sc in scans:
        r = strain_to_stress_propagated(
            sc, label, c11_gpa=c11_gpa, c12_gpa=c12_gpa, c44_gpa=c44_gpa,
            c11_err=c11_err, c12_err=c12_err, c44_err=c44_err, mode=mode,
            roi_mask=roi_mask)
        rows.append({
            "scan": getattr(sc, "name", "?"),
            "sigma_xx": round(r["sigma_xx"]["value"], 4), "sigma_xx_err": round(r["sigma_xx"]["std"], 4),
            "sigma_yy": round(r["sigma_yy"]["value"], 4), "sigma_yy_err": round(r["sigma_yy"]["std"], 4),
            "sigma_xy": round(r["sigma_xy"]["value"], 4), "sigma_xy_err": round(r["sigma_xy"]["std"], 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation / multivariate (channel correlation, PCA over scans)
# ─────────────────────────────────────────────────────────────────────────────

def channel_correlation(scan: Any, label: str = "without_roi",
                        roi_mask: "np.ndarray | None" = None):
    """Pixel-wise Pearson correlation between the 3 strain channels (3×3 DataFrame)."""
    import pandas as pd
    cols = {}
    for ch in CHANNELS:
        v = _channel_values(scan, label, ch, roi_mask, as_percent=False)
        cols[ch] = v
    lens = [c.size for c in cols.values() if c is not None]
    if not lens:
        return pd.DataFrame()
    n = min(lens)
    mat = np.vstack([cols[ch][:n] for ch in CHANNELS])
    corr = np.corrcoef(mat)
    return pd.DataFrame(corr, index=list(CHANNELS), columns=list(CHANNELS))


def _scan_feature_vector(scan: Any, label: str, roi_mask=None) -> "np.ndarray | None":
    """Per-scan feature vector = [mean,std,skew,kurtosis] × {eyy,exx,exy}."""
    stats = map_statistics(scan, label, roi_mask)
    if not stats:
        return None
    feats = []
    for ch in CHANNELS:
        s = stats.get(ch, {})
        feats += [s.get("mean", 0.0), s.get("std", 0.0),
                  s.get("skew", 0.0), s.get("kurtosis", 0.0)]
    return np.asarray(feats, dtype=float)


def pca_scans(scans: list, label: str = "without_roi", *, n_components: int = 2,
              roi_mask=None) -> dict:
    """PCA over per-scan feature vectors (scikit-learn). Robust to differing map shapes.

    Returns {names, coords (n_scans × k), explained_variance_ratio, feature_names}.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    names, X = [], []
    for sc in scans:
        fv = _scan_feature_vector(sc, label, roi_mask)
        if fv is not None:
            names.append(getattr(sc, "name", "?"))
            X.append(fv)
    if len(X) < 2:
        return {"names": names, "coords": np.zeros((len(X), 0)),
                "explained_variance_ratio": np.zeros(0), "feature_names": []}
    X = np.vstack(X)
    Xs = StandardScaler().fit_transform(X)
    k = int(min(n_components, Xs.shape[0], Xs.shape[1]))
    pca = PCA(n_components=k)
    coords = pca.fit_transform(Xs)
    feat_names = [f"{m}_{ch}" for ch in CHANNELS
                  for m in ("mean", "std", "skew", "kurt")]
    return {"names": names, "coords": coords,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "feature_names": feat_names}


def correlation_figure(scan: Any, label: str = "without_roi", roi_mask=None):
    """Heatmap of the 3×3 channel correlation matrix for one scan."""
    from matplotlib.figure import Figure
    corr = channel_correlation(scan, label, roi_mask)
    fig = Figure(figsize=(4, 3.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    if corr.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center"); ax.axis("off")
        return fig
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    labs = [CH_LABEL[c] for c in CHANNELS]
    ax.set_xticklabels(labs); ax.set_yticklabels(labs)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Channel correlation — {getattr(scan, 'name', '')}", fontsize=9)
    return fig


def pca_figure(scans: list, label: str = "without_roi", roi_mask=None):
    """Scatter of scans in PC1–PC2 space (clustering of experiments)."""
    from matplotlib.figure import Figure
    res = pca_scans(scans, label, n_components=2, roi_mask=roi_mask)
    fig = Figure(figsize=(5.5, 4.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    coords = res["coords"]
    if coords.shape[1] < 2:
        ax.text(0.5, 0.5, "need ≥2 scans for PCA", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.scatter(coords[:, 0], coords[:, 1], s=60, c="#1565C0", zorder=3)
    for name, (x, y) in zip(res["names"], coords[:, :2]):
        ax.annotate(name, (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    evr = res["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)" if len(evr) > 0 else "PC1")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)" if len(evr) > 1 else "PC2")
    ax.set_title("PCA — scans by strain-map features", fontsize=10)
    ax.grid(alpha=0.3)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_dataframe(df, path: str) -> str:
    """Write a DataFrame to .csv or .xlsx (by extension)."""
    from pathlib import Path
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(p, index=False)
    else:
        df.to_csv(p, index=False)
    return str(p)
