"""Prepare export assets: per-channel maps + full calib/report figures (no PIL crops)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

STRAIN_CHANNELS = ("exx", "eyy", "exy", "orientation")
STRESS_CHANNELS = ("sxx", "syy", "sxy")
LABELS = ("without_roi", "with_roi")
CALIB_KEYS = (
    "probe", "select6", "detection", "roi", "origin", "ellipse",
    "q_pixel", "basis", "indexing",
)

_LABEL_TITLE = {
    "without_roi": "Theoretical reference (without ROI)",
    "with_roi": "Experimental reference (with ROI)",
}
_CH_TITLE = {
    "exx": "ε_xx (%)", "eyy": "ε_yy (%)", "exy": "ε_xy (%)",
    "orientation": "θ (°)", "theta": "θ (°)",
    "sxx": "σ_xx", "syy": "σ_yy", "sxy": "σ_xy",
}
_CALIB_TITLE = {
    "probe": "Probe", "select6": "Detection @ 6 points", "detection": "Detection",
    "roi": "ROI", "origin": "Origin", "ellipse": "Ellipse",
    "q_pixel": "Q-pixel", "basis": "Basis", "indexing": "BVM indexing",
}


def _panel_png(arr: np.ndarray, path: Path, *, title: str, cmap: str,
               vmin: float | None, vmax: float | None, dpi: int = 150) -> dict[str, Any]:
    """Save one map panel with colorbar; return metadata for the manifest.

    When ``vmin``/``vmax`` are given (GUI vranges), all panels of that channel
    share the same clim — never per-array percentiles.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    a = np.asarray(arr, dtype=float)
    finite = a[np.isfinite(a)]
    fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    if finite.size == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
        ax.set_axis_off()
        vmin = vmax = 0.0
    else:
        if vmin is None or vmax is None:
            # Fallback only when GUI stress_vmax==0 (auto)
            pad = float(np.nanpercentile(np.abs(finite), 98)) or 1.0
            vmin, vmax = -pad, pad
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)  # no bbox_inches='tight' — keeps titles intact
    plt.close(fig)
    return {
        "path": str(path),
        "title": title,
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap,
        "shape": list(a.shape),
    }


def _savefig_figure(fig, path: Path, *, dpi: int = 150) -> bool:
    if fig is None or not hasattr(fig, "savefig"):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi)  # full figure — never crop
        return True
    except Exception:
        return False


def prepare_export_assets(
    scans: list,
    out_dir: Path | str,
    *,
    dpi: int = 150,
    include_maps: bool = True,
    include_calib: bool = True,
    include_reports: bool = True,
) -> dict:
    """Build ``export_assets/`` manifest + PNGs for the report writers.

    Map channels are redrawn from arrays (layout-agnostic). Calibration and
    ``report_*`` figures are saved whole — never PIL-split.
    """
    import engine as E

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": 2,
        "dpi": dpi,
        "created": datetime.now(timezone.utc).isoformat(),
        "options": {
            "include_maps": include_maps,
            "include_calib": include_calib,
            "include_reports": include_reports,
        },
        "scans": {},
    }

    for sc in scans:
        scan_entry: dict[str, Any] = {
            "name": sc.name,
            "labels": {},
            "calibrations": {},
            "reports": {},
        }
        cmap = str(getattr(sc.params, "strain_cmap", None) or "RdBu_r")

        if include_maps:
            for label in LABELS:
                lab_entry: dict[str, Any] = {
                    "title": _LABEL_TITLE[label], "channels": {},
                }
                for ch in STRAIN_CHANNELS:
                    arr = E.channel_map_2d(sc, ch, label)
                    if arr is None:
                        continue
                    rel = Path(sc.name) / f"strain_{label}" / f"{ch}.png"
                    clim = E.channel_clim(sc, ch)
                    vmin = clim[0] if clim else None
                    vmax = clim[1] if clim else None
                    ch_cmap = cmap
                    if ch in ("orientation", "theta"):
                        ch_cmap = str(getattr(sc.params, "strain_cmap_theta", None)
                                      or "PRGn")
                    meta = _panel_png(
                        arr, out_dir / rel,
                        title=f"{sc.name} — {_CH_TITLE[ch]} ({label})",
                        cmap=ch_cmap, vmin=vmin, vmax=vmax, dpi=dpi)
                    lab_entry["channels"][ch] = {
                        **meta,
                        "relpath": str(rel).replace("\\", "/"),
                        "kind": "strain",
                    }
                for ch in STRESS_CHANNELS:
                    arr = E.channel_map_2d(sc, ch, label)
                    if arr is None:
                        continue
                    rel = Path(sc.name) / f"stress_{label}" / f"{ch}.png"
                    clim = E.channel_clim(sc, ch)
                    vmin = clim[0] if clim else None
                    vmax = clim[1] if clim else None
                    meta = _panel_png(
                        arr, out_dir / rel,
                        title=f"{sc.name} — {_CH_TITLE[ch]} ({label})",
                        cmap=cmap, vmin=vmin, vmax=vmax, dpi=dpi)
                    lab_entry["channels"][ch] = {
                        **meta,
                        "relpath": str(rel).replace("\\", "/"),
                        "kind": "stress",
                    }
                if lab_entry["channels"]:
                    scan_entry["labels"][label] = lab_entry

        if include_calib:
            # Ensure Index BVM figure exists if indexing was run
            if (getattr(sc, "indexing_result", None) is not None
                    and E.resolve_figure(sc, "indexing") is None):
                try:
                    import bvm_indexing as bix
                    fig = bix.make_indexing_figure(
                        sc.indexing_result, title=f"{sc.name} — BVM indexing")
                    E.register_figure(sc, "indexing", fig, force=True)
                except Exception:
                    pass
            for key in CALIB_KEYS:
                fig = E.resolve_figure(sc, key)
                if fig is None:
                    continue
                rel = Path(sc.name) / "calib" / f"{key}.png"
                if _savefig_figure(fig, out_dir / rel, dpi=dpi):
                    title = f"{sc.name} — {_CALIB_TITLE.get(key, key)}"
                    scan_entry["calibrations"][key] = {
                        "relpath": str(rel).replace("\\", "/"),
                        "title": title,
                        "kind": "calib",
                    }

        if include_reports:
            for key in E.list_figure_keys(sc):
                if not str(key).startswith("report_"):
                    continue
                fig = E.resolve_figure(sc, key)
                if fig is None:
                    continue
                safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in key)
                rel = Path(sc.name) / "reports" / f"{safe}.png"
                if _savefig_figure(fig, out_dir / rel, dpi=dpi):
                    scan_entry["reports"][key] = {
                        "relpath": str(rel).replace("\\", "/"),
                        "title": f"{sc.name} — {key}",
                        "kind": "report",
                    }

        if (scan_entry["labels"] or scan_entry["calibrations"]
                or scan_entry["reports"]):
            manifest["scans"][sc.name] = scan_entry

    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(man_path)
    manifest["root"] = str(out_dir)
    return manifest


# Back-compat alias
def prepare_map_assets(scans: list, out_dir: Path | str, *, dpi: int = 150) -> dict:
    return prepare_export_assets(scans, out_dir, dpi=dpi)


def load_manifest(root: Path | str) -> dict:
    root = Path(root)
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data["root"] = str(root)
    return data


def iter_report_images(manifest: dict) -> list[dict]:
    """Ordered flat list for writers: calib → maps → reports, per scan."""
    root = Path(manifest["root"])
    rows: list[dict] = []
    for scan_name, sc in (manifest.get("scans") or {}).items():
        for key, meta in (sc.get("calibrations") or {}).items():
            path = root / (meta.get("relpath") or "")
            if path.is_file():
                rows.append({
                    "scan": scan_name,
                    "section": "Calibrations",
                    "label": "calib",
                    "label_title": "Calibrations",
                    "channel": key,
                    "kind": "calib",
                    "title": meta.get("title", key),
                    "path": path,
                })
        for label, lab in (sc.get("labels") or {}).items():
            for ch, meta in (lab.get("channels") or {}).items():
                path = root / (meta.get("relpath") or "")
                if path.is_file():
                    rows.append({
                        "scan": scan_name,
                        "section": "Maps",
                        "label": label,
                        "label_title": lab.get("title", label),
                        "channel": ch,
                        "kind": meta.get("kind", "strain"),
                        "title": meta.get("title", ch),
                        "path": path,
                    })
        for key, meta in (sc.get("reports") or {}).items():
            path = root / (meta.get("relpath") or "")
            if path.is_file():
                rows.append({
                    "scan": scan_name,
                    "section": "Reports",
                    "label": "reports",
                    "label_title": "Reports (user-sent)",
                    "channel": key,
                    "kind": "report",
                    "title": meta.get("title", key),
                    "path": path,
                })
    return rows


# Back-compat
iter_channel_images = iter_report_images
