"""Fast Batch Analysis — serial multi-scan pipeline + statistical line profiles.

Reads Parametros_cal.json, runs the Fast Analysis pipeline for each scan
using pre-computed braggpeaks.h5 files (skipping heavy disk detection),
then computes cross-scan statistics on line profiles.

Output layout::

    <batch_json_dir>/gui_results_fast_batch/
        <scan_stem>/          # per-scan artifacts (same as fast_artifacts)
            data/
            figures/
        batch_statistics/
            per_scan_profiles.csv
            batch_statistics.csv
            stats_<label>_<line>.png
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

FAST_BATCH_DIR = "gui_results_fast_batch"
BATCH_STATS_DIR = "batch_statistics"


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FastBatchScanConfig:
    name: str
    raw_path: str = ""
    braggpeaks_path: str = ""
    roi_bounds: list = field(default_factory=list)
    center_guess: list = field(default_factory=lambda: [128.0, 128.0])
    step10: dict = field(default_factory=dict)   # origin
    step11: dict = field(default_factory=dict)   # q_pixel
    step12: dict = field(default_factory=dict)   # basis
    step13: dict = field(default_factory=dict)   # strain
    stress_cfg: dict = field(default_factory=dict)
    line_positions: dict = field(default_factory=dict)  # {L1: [[x0,y0],[x1,y1]]}
    line_width: int = 1


@dataclass
class FastBatchConfig:
    source_path: Path
    scans: list[FastBatchScanConfig] = field(default_factory=list)
    options: dict = field(default_factory=dict)
    template_name: str | None = None
    fixed_line_profiles: dict = field(default_factory=dict)
    line_profiles_per_scan: dict = field(default_factory=dict)
    line_width: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_fast_batch_config(json_path: str | Path) -> FastBatchConfig:
    """Parse Parametros_cal.json → FastBatchConfig."""
    path = Path(json_path).expanduser()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    options = data.get("options", {})
    lw = int(options.get("line_width", 1))
    fixed_lp = data.get("fixed_line_profiles", {})
    lp_per_scan = data.get("line_profiles_per_scan", {})

    scans_dict = data.get("scans", {})
    template_idx = int(data.get("template_index", 0))
    scan_names = list(scans_dict.keys())
    template_name = (scan_names[template_idx]
                     if 0 <= template_idx < len(scan_names) else None)

    scans = []
    for name, sc in scans_dict.items():
        ovr = sc.get("step_overrides", {})
        # Use per-scan line positions if available, else fall back to global fixed lines
        lines = lp_per_scan.get(name, fixed_lp)
        scans.append(FastBatchScanConfig(
            name=name,
            raw_path=sc.get("raw_path", ""),
            braggpeaks_path=sc.get("braggpeaks_path", ""),
            roi_bounds=sc.get("roi_bounds", []),
            center_guess=sc.get("center_guess", [128.0, 128.0]),
            step10=ovr.get("step10", {}),
            step11=ovr.get("step11", {}),
            step12=ovr.get("step12", {}),
            step13=ovr.get("step13", {}),
            stress_cfg=ovr.get("stress", {}),
            line_positions=lines,
            line_width=lw,
        ))

    return FastBatchConfig(
        source_path=path,
        scans=scans,
        options=options,
        template_name=template_name,
        fixed_line_profiles=fixed_lp,
        line_profiles_per_scan=lp_per_scan,
        line_width=lw,
    )


def batch_summary_table(config: FastBatchConfig) -> list[dict]:
    """Return a list of dicts (one per scan) with key parameters for the cross-scan table."""
    rows = []
    for sc in config.scans:
        s11 = sc.step11
        s12 = sc.step12
        s13 = sc.step13
        cbv = s12.get("choose_basis_vectors", {})
        rows.append({
            "Scan": sc.name,
            "Q px (Å⁻¹/px)": s11.get("px_guess", ""),
            "k_max": s11.get("kmax", ""),
            "QR rot (°)": s12.get("qr_rotation", ""),
            "QR flip": s12.get("qr_flip", False),
            "Manual basis": s12.get("manual_enabled", False),
            "Origin idx": cbv.get("index_origin", 0),
            "g1 idx": cbv.get("index_g1", 0),
            "g2 idx": cbv.get("index_g2", 0),
            "Rot strain (°)": s13.get("coordinate_rotation", ""),
            "vrange": s13.get("vrange", ""),
            "C11 (GPa)": sc.stress_cfg.get("c11_gpa", ""),
            "Lines": ", ".join(sc.line_positions.keys()),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def _L(log: Callable | None, msg: str) -> None:
    if log:
        try:
            log(msg)
        except Exception:
            pass


def _apply_calibration(state: Any, scan: FastBatchScanConfig, log=None) -> None:
    """Apply all lightweight calibration parameters from the JSON config to state."""
    try:
        from pipeline import (
            set_roi_from_bounds, set_origin_center_guess,
            set_q_pixel_size_step, update_strain_basis_params,
            setup_basis_step, update_strain_params,
            set_strain_scan_roi_from_bounds,
        )
    except ImportError:
        from .pipeline import (
            set_roi_from_bounds, set_origin_center_guess,
            set_q_pixel_size_step, update_strain_basis_params,
            setup_basis_step, update_strain_params,
            set_strain_scan_roi_from_bounds,
        )

    # ROI (calibration region for BVM / q-pixel fit)
    if scan.roi_bounds:
        try:
            set_roi_from_bounds(state, scan.roi_bounds, log)
        except Exception as e:
            _L(log, f"  [warn] ROI: {e}")

    # Origin / center guess
    if scan.center_guess:
        try:
            s10 = scan.step10
            set_origin_center_guess(state, tuple(scan.center_guess),
                                    sampling=int(s10.get("sampling", 2)), log=log)
        except Exception as e:
            _L(log, f"  [warn] center guess: {e}")

    # Q pixel size (direct value in Å⁻¹/px)
    s11 = scan.step11
    if s11:
        try:
            set_q_pixel_size_step(state,
                                  q_pixel_size=float(s11.get("px_guess", 0.01)),
                                  units="A^-1", log=log)
        except Exception as e:
            _L(log, f"  [warn] Q pixel: {e}")

    # Basis vector parameters
    s12 = scan.step12
    if s12:
        cbv = s12.get("choose_basis_vectors", {})
        vp = cbv.get("vis_params", {})
        try:
            update_strain_basis_params(
                state,
                min_spacing=int(cbv.get("minSpacing", 20)),
                min_absolute_intensity=int(cbv.get("minAbsoluteIntensity", 80)),
                max_num_peaks=int(cbv.get("maxNumPeaks", 60)),
                edge_boundary=int(cbv.get("edgeBoundary", 4)),
                vmin=float(vp.get("vmin", 0.0)),
                vmax=float(vp.get("vmax", 0.995)),
                qr_rotation=float(s12.get("qr_rotation", 0.0)),
                qr_flip=bool(s12.get("qr_flip", False)),
                manual_enabled=bool(s12.get("manual_enabled", False)),
                index_origin=int(cbv.get("index_origin", 0)),
                index_g1=int(cbv.get("index_g1", 0)),
                index_g2=int(cbv.get("index_g2", 0)),
                log=log,
            )
        except Exception as e:
            _L(log, f"  [warn] basis params: {e}")

        try:
            setup_basis_step(state, log)
        except Exception as e:
            _L(log, f"  [warn] setup_basis_step: {e}")

    # Strain parameters
    s13 = scan.step13
    if s13:
        try:
            vrange = s13.get("vrange", [-5, 5])
            vrange_theta = s13.get("vrange_theta", vrange)
            update_strain_params(
                state,
                coordinate_rotation=float(s13.get("coordinate_rotation", 0)),
                max_peak_spacing=float(s13.get("max_peak_spacing", 2.0)),
                layout=str(s13.get("layout", "horizontal")),
                vrange=vrange,
                vrange_theta=vrange_theta,
                cmap=str(s13.get("cmap", "RdBu_r")),
                cmap_theta=str(s13.get("cmap_theta", "PRGn")),
                show_orientation=bool(s13.get("show_orientation", True)),
                log=log,
            )
        except Exception as e:
            _L(log, f"  [warn] strain params: {e}")

        if "scan_roi_bounds" in s13:
            try:
                set_strain_scan_roi_from_bounds(state, s13["scan_roi_bounds"], log)
            except Exception as e:
                _L(log, f"  [warn] strain scan ROI: {e}")


def _run_strain(state: Any, options: dict, log=None) -> list[str]:
    """Run get_strain for requested combinations. Returns list of computed labels."""
    try:
        from pipeline import compute_strain_map_step
    except ImportError:
        from .pipeline import compute_strain_map_step

    computed: list[str] = []

    if options.get("do_strain_wo", True):
        try:
            compute_strain_map_step(state, use_roi=False,
                                    label_override="without_roi", log=log)
            computed.append("without_roi")
        except Exception as e:
            _L(log, f"  [ERROR] strain without ROI: {e}")

    if options.get("do_strain_roi", False):
        try:
            compute_strain_map_step(state, use_roi=True,
                                    label_override="with_roi", log=log)
            computed.append("with_roi")
        except Exception as e:
            _L(log, f"  [ERROR] strain with ROI: {e}")

    return computed


def _run_stress(state: Any, scan: FastBatchScanConfig,
                computed_labels: list[str], options: dict, log=None) -> None:
    """Compute stress for each available strain label."""
    try:
        from pipeline import compute_stress_analysis_step
    except ImportError:
        from .pipeline import compute_stress_analysis_step

    sc = scan.stress_cfg
    if not sc:
        return

    mode = sc.get("mode", "plane_stress")
    c11 = float(sc.get("c11_gpa", 165.7)) * 1e9   # GPa → Pa
    c12 = float(sc.get("c12_gpa", 63.9)) * 1e9
    c44 = float(sc.get("c44_gpa", 79.6)) * 1e9

    for label in computed_labels:
        try:
            compute_stress_analysis_step(state, label=label,
                                         mode=mode, c11_pa=c11, c12_pa=c12, c44_pa=c44,
                                         log=log)
        except Exception as e:
            _L(log, f"  [warn] stress {label}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Line profile extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_line_profiles(strain_raw: dict, adf: "np.ndarray | None",
                           lines: dict, line_width: int = 1) -> dict:
    """Extract strain + ADF profiles along each defined line.

    lines: {L1: [[x0,y0],[x1,y1]], ...}  (x=col, y=row)

    Returns: {
        label: {line_id: {eyy, exx, exy, dist_px}},
        "adf":  {line_id: {intensity, dist_px}},
    }
    """
    try:
        from fast_artifacts import _as_hw3
    except ImportError:
        from .fast_artifacts import _as_hw3

    def _profile_1d(arr2d: np.ndarray, x0: float, y0: float,
                    x1: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns (dist_px, values) or (empty, empty) on failure."""
        try:
            from skimage.measure import profile_line
            prof = profile_line(arr2d, (y0, x0), (y1, x1),
                                linewidth=line_width, order=1, mode="nearest")
        except ImportError:
            try:
                from scipy.ndimage import map_coordinates
                n = max(2, int(np.hypot(x1 - x0, y1 - y0)) + 1)
                rows = np.linspace(y0, y1, n)
                cols = np.linspace(x0, x1, n)
                prof = map_coordinates(arr2d, [rows, cols], order=1, mode="nearest")
            except Exception:
                return np.array([]), np.array([])
        dist = np.linspace(0.0, np.hypot(x1 - x0, y1 - y0), len(prof))
        return dist, prof

    def _parse_coords(ldef):
        try:
            p0, p1 = ldef
            return float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])
        except Exception:
            return None

    results: dict = {}

    # Strain channels
    for label, raw in (strain_raw or {}).items():
        hw3 = _as_hw3(raw)
        if hw3 is None:
            continue
        results[label] = {}
        for line_id, ldef in lines.items():
            coords = _parse_coords(ldef)
            if coords is None:
                continue
            x0, y0, x1, y1 = coords
            entry: dict = {}
            for ch_i, ch in enumerate(("eyy", "exx", "exy")):
                dist, prof = _profile_1d(hw3[..., ch_i], x0, y0, x1, y1)
                if dist.size:
                    entry[ch] = prof
                    entry["dist_px"] = dist
            if entry:
                results[label][line_id] = entry

    # ADF intensity
    if adf is not None:
        results["adf"] = {}
        for line_id, ldef in lines.items():
            coords = _parse_coords(ldef)
            if coords is None:
                continue
            x0, y0, x1, y1 = coords
            dist, prof = _profile_1d(adf, x0, y0, x1, y1)
            if dist.size:
                results["adf"][line_id] = {"intensity": prof, "dist_px": dist}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Single-scan runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_scan(scan: FastBatchScanConfig, output_root: Path,
                  options: dict, log=None) -> dict:
    """Full pipeline for one scan: load → calibrate → strain → save → profiles."""
    try:
        from state import WorkflowState
        from pipeline import load_braggpeaks_file, try_load_adf_from_sidecar_h5
        from fast_artifacts import _adf_from_state, save_fast_artifacts
    except ImportError:
        from .state import WorkflowState
        from .pipeline import load_braggpeaks_file, try_load_adf_from_sidecar_h5
        from .fast_artifacts import _adf_from_state, save_fast_artifacts

    _L(log, f"\n{'─' * 60}")
    _L(log, f"  Scan: {scan.name}")

    state = WorkflowState()
    state.raw_mib_path = scan.raw_path

    # 1. Load pre-computed braggpeaks (fast — avoids disk detection)
    _L(log, "  → Loading braggpeaks…")
    try:
        braggpeaks = load_braggpeaks_file(scan.braggpeaks_path, log=log)
        state.braggpeaks = braggpeaks
        state.braggpeaks_path = scan.braggpeaks_path
    except Exception as e:
        _L(log, f"  [ERROR] Cannot load braggpeaks: {e}")
        return {"error": str(e), "profiles": {}, "scan_name": scan.name}

    # 2. Load ADF from sidecar h5 (returns ndarray or None)
    try:
        adf_arr = try_load_adf_from_sidecar_h5(scan.raw_path, log=log)
        if adf_arr is not None:
            state.virtual_images = {"annular_dark_field": adf_arr}
    except Exception:
        pass

    # 3. Apply per-scan calibration (lightweight)
    _L(log, "  → Applying calibration…")
    _apply_calibration(state, scan, log)

    # 4. Strain computation (HEAVY — calls py4DSTEM get_strain)
    _L(log, "  → Computing strain (heavy)…")
    computed = _run_strain(state, options, log)
    if not computed:
        _L(log, "  [warn] No strain maps computed — check calibration params")

    # 5. Stress (light — Hooke over saved ε)
    if any(options.get(k) for k in ("do_stress_wo", "do_stress_roi")):
        _L(log, "  → Computing stress…")
        _run_stress(state, scan, computed, options, log)

    # 6. Save per-scan artifacts
    _L(log, "  → Saving artifacts…")
    try:
        save_fast_artifacts(state, output_root=output_root,
                            save_figures=False, log=log)
    except Exception as e:
        _L(log, f"  [warn] Save artifacts: {e}")

    # 7. Extract line profiles (light)
    _L(log, "  → Extracting line profiles…")
    adf = _adf_from_state(state)
    strain_raw = getattr(state, "strain_raw", {}) or {}
    profiles = _extract_line_profiles(strain_raw, adf, scan.line_positions, scan.line_width)

    _L(log, f"  Done: {len(profiles)} label(s), {len(scan.line_positions)} line(s)")
    return {"profiles": profiles, "scan_name": scan.name}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-scan statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_batch_statistics(scan_results: dict) -> dict:
    """Compute mean / std / SEM across scans for every label / line / channel.

    scan_results: {scan_name: {"profiles": {label: {line_id: {ch: arr, dist_px: arr}}}}}

    Returns: {label: {line_id: {ch: {mean, std, sem, n, min, max, dist_px, individual}}}}
    """
    stats: dict = {}

    # Collect all labels
    all_labels: set = set()
    for r in scan_results.values():
        all_labels.update((r.get("profiles") or {}).keys())

    for label in all_labels:
        stats[label] = {}
        all_lines: set = set()
        for r in scan_results.values():
            all_lines.update(
                ((r.get("profiles") or {}).get(label) or {}).keys())

        for line_id in all_lines:
            stats[label][line_id] = {}
            all_ch: set = set()
            for r in scan_results.values():
                d = (((r.get("profiles") or {}).get(label) or {})
                     .get(line_id) or {})
                all_ch.update(k for k in d if k != "dist_px")

            for ch in all_ch:
                profiles: list[np.ndarray] = []
                dist_ref: np.ndarray | None = None
                for r in scan_results.values():
                    d = (((r.get("profiles") or {}).get(label) or {})
                         .get(line_id) or {})
                    p = d.get(ch)
                    if p is not None and len(p) > 0:
                        profiles.append(np.asarray(p, dtype=float))
                        if dist_ref is None:
                            dist_ref = d.get("dist_px")

                if not profiles:
                    continue

                min_len = min(len(p) for p in profiles)
                aligned = np.stack([p[:min_len] for p in profiles], axis=0)
                n = aligned.shape[0]
                dist = (dist_ref[:min_len] if dist_ref is not None
                        else np.arange(min_len, dtype=float))

                stats[label][line_id][ch] = {
                    "mean":       np.mean(aligned, axis=0),
                    "std":        np.std(aligned, axis=0),
                    "sem":        np.std(aligned, axis=0) / max(np.sqrt(n), 1),
                    "n":          n,
                    "min":        np.min(aligned, axis=0),
                    "max":        np.max(aligned, axis=0),
                    "dist_px":    dist,
                    "individual": aligned,   # (n_scans, n_points) — for figures
                }

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

_CH_LABELS = {"eyy": "εyy", "exx": "εxx", "exy": "εxy", "intensity": "ADF"}


def export_batch_results(config: FastBatchConfig, scan_results: dict,
                          stats: dict, output_dir: str | Path,
                          log=None) -> list[Path]:
    """Save:
    - per_scan_profiles.csv   (long format: scan, label, line, channel, dist, value)
    - batch_statistics.csv    (mean / std / sem / min / max per position)
    - stats_<label>_<line>.png (individual + mean±std figure per line)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import pandas as pd
        _has_pd = True
    except ImportError:
        _has_pd = False
        _L(log, "  [warn] pandas not installed — CSV export skipped")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ── per-scan profiles CSV ─────────────────────────────────────────────────
    if _has_pd:
        rows: list[dict] = []
        for scan_name, result in scan_results.items():
            profiles = result.get("profiles") or {}
            for label, ldata in profiles.items():
                for line_id, line_data in ldata.items():
                    dist = line_data.get("dist_px", np.array([]))
                    for ch in ("eyy", "exx", "exy", "intensity"):
                        vals = line_data.get(ch)
                        if vals is None:
                            continue
                        n = min(len(dist), len(vals))
                        for i in range(n):
                            rows.append({
                                "scan": scan_name, "label": label,
                                "line": line_id, "channel": ch,
                                "dist_px": round(float(dist[i]), 3),
                                "value": round(float(vals[i]), 6),
                            })
        if rows:
            p = output_dir / "per_scan_profiles.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            written.append(p)
            _L(log, f"  saved {p.name}  ({len(rows)} rows)")

    # ── batch statistics CSV ──────────────────────────────────────────────────
    if _has_pd:
        stat_rows: list[dict] = []
        for label, ldata in stats.items():
            for line_id, cdata in ldata.items():
                for ch, cs in cdata.items():
                    dist = cs["dist_px"]
                    mean = cs["mean"]
                    std  = cs["std"]
                    sem  = cs["sem"]
                    mn   = cs["min"]
                    mx   = cs["max"]
                    for i in range(len(mean)):
                        stat_rows.append({
                            "label": label, "line": line_id,
                            "channel": ch, "n_scans": cs["n"],
                            "dist_px": round(float(dist[i]), 3),
                            "mean": round(float(mean[i]), 6),
                            "std":  round(float(std[i]),  6),
                            "sem":  round(float(sem[i]),  6),
                            "min":  round(float(mn[i]),   6),
                            "max":  round(float(mx[i]),   6),
                        })
        if stat_rows:
            p = output_dir / "batch_statistics.csv"
            pd.DataFrame(stat_rows).to_csv(p, index=False)
            written.append(p)
            _L(log, f"  saved {p.name}  ({len(stat_rows)} rows)")

    # ── figures: per label + line ─────────────────────────────────────────────
    scan_names = list(scan_results.keys())
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for label, ldata in stats.items():
        for line_id, cdata in ldata.items():
            channels = [c for c in ("eyy", "exx", "exy") if c in cdata]
            if not channels:
                continue

            fig, axes = plt.subplots(1, len(channels),
                                     figsize=(5 * len(channels), 4), dpi=130,
                                     squeeze=False)
            axes = axes[0]

            for ax, ch in zip(axes, channels):
                cs = cdata[ch]
                dist = cs["dist_px"]
                indiv = cs.get("individual")   # (n_scans, n_pts)

                # Individual scan profiles
                if indiv is not None:
                    for i, row in enumerate(indiv):
                        c = prop_colors[i % len(prop_colors)]
                        lbl = scan_names[i] if i < len(scan_names) else f"S{i+1}"
                        ax.plot(dist[:len(row)], row * 100, color=c,
                                alpha=0.4, lw=0.9,
                                label=lbl if i < 6 else None)

                # Mean ± std band
                ax.plot(dist, cs["mean"] * 100, "k-", lw=2.0,
                        label="Mean", zorder=5)
                ax.fill_between(dist,
                                (cs["mean"] - cs["std"]) * 100,
                                (cs["mean"] + cs["std"]) * 100,
                                alpha=0.20, color="black", label="±1σ", zorder=4)

                ax.set_xlabel("Distance (px)", fontsize=9)
                ax.set_ylabel(f"{_CH_LABELS.get(ch, ch)} (%)", fontsize=9)
                ax.set_title(f"{line_id} — {_CH_LABELS.get(ch, ch)}", fontsize=9)
                ax.tick_params(labelsize=8)
                if len(scan_names) <= 7:
                    ax.legend(fontsize=6, loc="upper right", ncol=1)

            n_sc = list(cdata.values())[0]["n"] if cdata else "?"
            fig.suptitle(f"Batch: {label} | {line_id} | n = {n_sc} scans",
                         fontsize=10, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))

            safe = label.replace("/", "_").replace(" ", "_")
            p = output_dir / f"stats_{safe}_{line_id}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(p)
            _L(log, f"  saved {p.name}")

    _L(log, f"\nExport complete → {output_dir}  ({len(written)} files)")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fast_batch_serial(
    config: FastBatchConfig,
    output_root: str | Path | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    scan_done_cb: Callable[[str, dict], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Process all scans in the batch serially.

    progress_cb(i, total, scan_name) is called before each scan and once at end.
    scan_done_cb(scan_name, result) is called immediately after each scan completes.

    Returns: {
        "scan_results":  {scan_name: {"profiles": ..., "error": ...}},
        "stats":         cross-scan statistics dict,
        "export_paths":  list of Path,
        "output_dir":    Path to batch_statistics dir,
    }
    """
    if output_root is None:
        output_root = config.source_path.parent / FAST_BATCH_DIR
    output_root = Path(output_root)

    stats_dir = output_root / BATCH_STATS_DIR
    n = len(config.scans)
    scan_results: dict = {}

    _L(log, f"Fast Batch Analysis — {n} scans")
    _L(log, f"Output root: {output_root}")

    for i, scan in enumerate(config.scans):
        if progress_cb:
            try:
                progress_cb(i, n, scan.name)
            except Exception:
                pass

        result = _run_one_scan(scan, output_root, config.options, log)
        scan_results[scan.name] = result

        if scan_done_cb:
            try:
                scan_done_cb(scan.name, result)
            except Exception:
                pass

    if progress_cb:
        try:
            progress_cb(n, n, "Computing statistics…")
        except Exception:
            pass

    _L(log, "\nComputing cross-scan statistics…")
    stats = compute_batch_statistics(scan_results)

    _L(log, "Exporting batch results…")
    try:
        export_paths = export_batch_results(
            config, scan_results, stats, stats_dir, log)
    except Exception as e:
        _L(log, f"[ERROR] Export failed: {e}")
        export_paths = []

    return {
        "scan_results": scan_results,
        "stats":        stats,
        "export_paths": export_paths,
        "output_dir":   stats_dir,
    }
