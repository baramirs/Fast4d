"""fast4d.driver — single / multi serial orchestration over the engine.

There is **no "mode"**. One scan and N scans take the *same* code path: a serial
loop calling the same engine steps. "Single" is just "N == 1". The GUI fills the
same calibration fields either way (one column vs many); the driver runs them
one after another.

Per scan, the driver auto-detects the entry path (the user's two workflows):

    Path A  — braggpeaks.h5 already exists (the common, light case):
        load_braggpeaks → calibrate → strain → save
    Path B  — raw data, no braggpeaks yet (first pass, OPTIONAL but needed):
        load_datacube → probe → compute_braggpeaks(.h5) → calibrate → strain → save

The interactive *tuning* of Path B (pick 6 ADF points, preview detection, dial in
detect_params) happens in the GUI before Compute; by the time the driver runs, the
tuned ``detect_params`` already live on each ``scan.params``. The driver only runs
the heavy full-scan detection.

COMPUTE (this module's ``compute_*``) is heavy and persisted. ANALYSIS
(``analyze_*``) is light and reads the saved strain arrays — never recomputes.
This mirrors the notebook's compute/analyze split and the engine's contract.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import engine as E

# Re-use the canonical Log type and _log helper from engine — single definition.
Log = E.Log
_log = E._log


class _Cancelled(Exception):
    """Internal: raised at a step boundary when the user cancels mid-scan."""


def _fmt_elapsed(seconds: float) -> str:
    """Human elapsed time (the Phase-1 timer convention): h:mm:ss or m:ss."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Progress reporting (GUI-agnostic). The GUI passes a callback; a script can omit.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProgressEvent:
    scan_index: int          # 0-based index of the scan in the batch
    n_scans: int             # total scans in this run (1 == "single")
    scan_name: str
    path: str                # "A" | "B" | ""
    step: str                # "load_braggpeaks" | "calibrate" | "strain:with_roi" | …
    message: str = ""        # short human note


Progress = Callable[[ProgressEvent], None] | None


def _emit(progress: Progress, ev: ProgressEvent) -> None:
    if progress is not None:
        try:
            progress(ev)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComputeOptions:
    """What COMPUTE should do per scan (same for single and multi)."""
    # calibration strategy:
    #   "fit"   → run_calibration_sequence (refits origin/q-pixel, per-step figures).
    #             The canonical first compute (multi fits each file independently).
    #   "apply" → apply_calibration (bulk re-apply of already-known values, no fit,
    #             no figures). Fast re-run of a fully-specified template.
    calibration: str = "fit"            # "fit" | "apply"
    do_without_roi: bool = True         # strain over the whole scan
    do_with_roi: bool = True            # strain restricted to the calibration ROI
    do_stress_without_roi: bool = False  # also derive the Hooke stress on without_roi
    do_stress_with_roi: bool = False     # also derive the Hooke stress on with_roi
    stress_mode: str = "plane_stress"    # "plane_stress" | "plane_strain"
    make_figures: bool = True           # per-step calibration figures (Report)
    figure_mode: str = "report"         # "off" | "preview" | "report" (see engine.FigurePolicy)
    max_figures_in_ram: int = 12        # per-scan registered figure cap
    close_orphan_pyplot: bool = True    # close stray matplotlib windows after previews
    store_figure: dict = field(default_factory=lambda: dict(E.DEFAULT_STORE_FIGURE))
    spill_to_disk: bool = True          # evicted in-RAM figures → PNG sidecar
    spill_dpi: int = 72                 # sidecar / temp PNG resolution (GUI viewing)
    save_figures_dpi: int = 150         # figures/ export on Compute / Save
    save: bool = True                   # persist workspace (npz + manifest)
    save_figures: bool = True           # also dump per-step PNGs for the Report
    output_root: str | None = None      # where save_results writes (None = default)
    save_braggpeaks: bool = True        # Path B: persist the computed braggpeaks.h5
    stop_on_error: bool = False         # multi: keep going past a failed scan
    vimg_cmap: str = "gray"             # colormap for the saved virtual-image PNGs


@dataclass
class AnalyzeOptions:
    """Light per-scan analysis over saved strain arrays (no get_strain)."""
    do_stress: bool = True
    stress_mode: str = "plane_stress"             # "plane_stress" | "plane_strain"
    labels: tuple = ("without_roi", "with_roi")   # which saved strain maps to use
    lines: dict = field(default_factory=dict)     # {line_id: [[x0,y0],[x1,y1]]}
    line_width: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Path detection & validation
# ─────────────────────────────────────────────────────────────────────────────

def detect_path(scan: E.Scan) -> str:
    """'A' when a usable braggpeaks.h5 is present; otherwise 'B' (build it)."""
    return E.analysis_path(scan)


def validate_scan(scan: E.Scan, opts: ComputeOptions | None = None) -> list[str]:
    """Pre-flight check; returns a list of human problems ([] == ready)."""
    problems: list[str] = []
    path = E.analysis_path(scan)
    if path == "B":
        if not scan.raw_path or not Path(scan.raw_path).is_file():
            problems.append("raw data file missing (needed to build braggpeaks)")
        src = (scan.params.probe_source or "vacuum").lower()
        if src == "vacuum" and not (scan.vacuum_path and Path(scan.vacuum_path).is_file()):
            problems.append("vacuum file required for the 'vacuum' probe source")
    return problems


def _braggpeaks_save_path(scan: E.Scan, opts: ComputeOptions) -> str:
    """Where to write a Path-B braggpeaks.h5 if the scan doesn't already name one."""
    if scan.braggpeaks_path:
        return scan.braggpeaks_path
    if opts.output_root:
        base = Path(opts.output_root)
    elif scan.raw_path:
        base = Path(scan.raw_path).parent
    else:
        base = Path.cwd()
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in scan.name) or "scan"
    return str(base / f"{safe}_braggpeaks.h5")


# ─────────────────────────────────────────────────────────────────────────────
# Outcomes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanOutcome:
    scan: E.Scan
    ok: bool
    path: str
    labels: list = field(default_factory=list)   # strain labels actually computed
    results_dir: str = ""
    error: str = ""
    elapsed_s: float = 0.0


@dataclass
class BatchOutcome:
    outcomes: list = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def n_ok(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for o in self.outcomes if not o.ok)

    def summary(self) -> str:
        return (f"{self.n_ok} ok / {self.n_failed} failed "
                f"in {_fmt_elapsed(self.elapsed_s)}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE  (heavy, persisted)
# ─────────────────────────────────────────────────────────────────────────────

def compute_scan(scan: E.Scan, opts: ComputeOptions | None = None, *,
                 log: Log = None, progress: Progress = None,
                 scan_index: int = 0, n_scans: int = 1,
                 cancel: Callable[[], bool] | None = None) -> ScanOutcome:
    """Full COMPUTE for ONE scan, auto-branching Path A vs Path B.

    Never raises: any failure is captured into ``ScanOutcome.error`` so a multi
    run can continue. Sets ``scan.status`` to computed / done / error / pending.
    ``cancel()`` is polled at every step boundary so a mid-scan cancel stops at the
    next step (a py4DSTEM/CUDA call already in flight isn't interruptible — that one
    finishes, then we bail).
    """
    opts = opts or ComputeOptions()
    t0 = time.perf_counter()
    path = E.analysis_path(scan)
    out = ScanOutcome(scan=scan, ok=False, path=path)

    E.set_figure_policy(
        mode=opts.figure_mode,
        store=dict(opts.store_figure),
        max_in_ram=opts.max_figures_in_ram,
        close_orphans=opts.close_orphan_pyplot,
        spill_to_disk=opts.spill_to_disk,
        spill_dpi=opts.spill_dpi,
        save_dpi=opts.save_figures_dpi,
    )
    eff_make_figures = bool(opts.make_figures and opts.figure_mode == "report")

    def emit(step: str, message: str = "") -> None:
        _emit(progress, ProgressEvent(scan_index, n_scans, scan.name, path, step, message))

    def ck() -> None:
        if cancel is not None and cancel():
            raise _Cancelled()

    try:
        problems = validate_scan(scan, opts)
        if problems:
            raise RuntimeError("; ".join(problems))
        ck()

        # ── get braggpeaks into the state ──────────────────────────────────────
        if path == "B":
            emit("load_datacube", "loading raw 4D datacube (heavy)")
            E.load_datacube(scan, log=log); ck()
            emit("probe", f"computing probe ({scan.params.probe_source})")
            E.compute_probe(scan, log=log); ck()
            emit("braggpeaks", "full-scan Bragg disk detection (CUDA)")
            save_bp = _braggpeaks_save_path(scan, opts) if opts.save_braggpeaks else None
            E.compute_braggpeaks(scan, save_path=save_bp, log=log); ck()
        else:
            emit("load_braggpeaks", "loading braggpeaks.h5")
            E.load_braggpeaks(scan, log=log)
            try:                       # ADF for overlays (best-effort, sidecar h5)
                E.load_adf(scan, log=log)
            except Exception:
                pass
            ck()

        # ── calibration (ROI → origin → ellipse → q-pixel → basis) ─────────────
        emit("calibrate", f"calibration ({opts.calibration})")
        if opts.calibration == "apply":
            E.apply_calibration(scan, log=log)
            emit("calibrate:apply", "calibration values applied")
        else:
            E.run_calibration_sequence(
                scan, make_figures=eff_make_figures, log=log,
                progress_step=lambda step: emit(f"calibrate:{step}", f"{step} done"))
        ck()

        # ── strain (heavy) ─────────────────────────────────────────────────────
        labels: list[str] = []
        if opts.do_without_roi:
            emit("strain:without_roi", "strain map — full scan (heavy)")
            E.compute_strain(scan, use_roi=False, log=log)
            labels.append("without_roi")
            ck()
        if opts.do_with_roi:
            if scan.params.roi_bounds or scan.params.strain_scan_roi_bounds:
                emit("strain:with_roi", "strain map — ROI as g1,g2 reference (heavy)")
                E.compute_strain(scan, use_roi=True, log=log)
                labels.append("with_roi")
                ck()
            else:
                _log(log, f"[{scan.name}] with-ROI strain skipped (no ROI / strain ROI set)")
        out.labels = labels

        # ── stress (light, Hooke's law from the strain just computed) ───────────
        stress_flags = {"without_roi": opts.do_stress_without_roi,
                        "with_roi": opts.do_stress_with_roi}
        for label in labels:
            if stress_flags.get(label):
                emit(f"stress:{label}", "stress map (Hooke's law)")
                try:
                    E.compute_stress(scan, label=label, mode=opts.stress_mode, log=log)
                except Exception as exc:
                    _log(log, f"[{scan.name}] stress ({label}) failed: {exc}")
                ck()

        # ── persist ────────────────────────────────────────────────────────────
        if opts.save:
            emit("save", "saving workspace + figures")
            E.save_results(scan, save_figures=opts.save_figures,
                           output_root=opts.output_root,
                           vimg_cmap=getattr(opts, "vimg_cmap", "gray"), log=log)
            out.results_dir = scan.results_dir
            # also dump the per-step calibration figures (origin/ellipse/q/basis…)
            # next to the workspace so the Report can load them later.
            if opts.save_figures and scan.results_dir:
                try:
                    E.save_figures(scan, Path(scan.results_dir) / "figures",
                                   dpi=opts.save_figures_dpi, log=log)
                except Exception as exc:
                    _log(log, f"[{scan.name}] figure dump skipped: {exc}")
            E.clean_duplicate_figure_pngs(scan, log=log)
            scan.status = "done"
        else:
            scan.status = "computed"

        out.ok = True
        emit("done", "ok")

    except _Cancelled:
        scan.status = "pending"            # cancelled is NOT an error
        out.error = "cancelled"
        emit("cancelled", "cancelled by user")
        _log(log, f"[{scan.name}] cancelled at a step boundary.")
    except Exception as exc:
        scan.status = "error"
        out.error = f"{type(exc).__name__}: {exc}"
        emit("error", out.error)
        _log(log, f"[{scan.name}] COMPUTE FAILED: {out.error}\n{traceback.format_exc()}")

    out.elapsed_s = time.perf_counter() - t0
    return out


def compute_all(scans: list[E.Scan], opts: ComputeOptions | None = None, *,
                log: Log = None, progress: Progress = None,
                on_scan_done: Callable[[ScanOutcome], None] | None = None,
                cancel: Callable[[], bool] | None = None) -> BatchOutcome:
    """Serial COMPUTE over N scans (N == 1 is just "single"). Times the whole run.

    ``on_scan_done(outcome)`` fires after each scan so the GUI can update its
    grid / loader incrementally (Phase-1 compute-all-then-loader behavior).
    ``cancel()`` is polled before each scan — a cooperative stop (the current
    scan always finishes; py4DSTEM steps aren't interruptible mid-call).
    """
    opts = opts or ComputeOptions()
    t0 = time.perf_counter()
    result = BatchOutcome()
    n = len(scans)
    _log(log, f"COMPUTE: {n} scan(s) — calibration={opts.calibration}, "
              f"strain wo={opts.do_without_roi} roi={opts.do_with_roi}")
    for i, scan in enumerate(scans):
        if cancel is not None and cancel():
            _log(log, f"CANCELLED — stopped before '{scan.name}' ({i}/{n} done).")
            break
        out = compute_scan(scan, opts, log=log, progress=progress,
                           scan_index=i, n_scans=n, cancel=cancel)
        result.outcomes.append(out)
        _log(log, f"[{scan.name}] {'OK' if out.ok else 'FAILED'} "
                  f"({_fmt_elapsed(out.elapsed_s)})"
                  + (f" — {out.error}" if out.error else ""))
        if on_scan_done is not None:
            try:
                on_scan_done(out)
            except Exception:
                pass
        # Release this scan's heavy buffers now that it's saved/reported — mirrors
        # the manual "Free RAM" button. Figures and ADF cache survive so the
        # on_scan_done GUI update above already had everything it needs. Also drop
        # the detected peaks IF they are persisted to disk (Path A): calibration and
        # strain for this scan are done, and ensure_braggpeaks_for_calibration will
        # re-load them from the .h5 on demand — so keeping them resident only wastes
        # RAM across a multi-scan batch.
        bp_path = getattr(scan, "braggpeaks_path", None)
        drop_bp = bool(bp_path) and Path(str(bp_path)).exists()
        E.free_memory([scan], drop_braggpeaks=drop_bp, log=log)
        if out.error == "cancelled" or (cancel is not None and cancel()):
            _log(log, f"CANCELLED — stopped after '{scan.name}' ({i + 1}/{n}).")
            break
        if not out.ok and opts.stop_on_error:
            _log(log, f"Stopping batch after failure on '{scan.name}'.")
            break
    result.elapsed_s = time.perf_counter() - t0
    _log(log, f"COMPUTE complete: {result.summary()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS  (light, reads saved strain arrays — never recomputes)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_scan(scan: E.Scan, opts: AnalyzeOptions | None = None, *,
                 log: Log = None) -> dict:
    """Stress (Hooke) + line profiles for ONE scan from its saved strain maps."""
    opts = opts or AnalyzeOptions()
    st = scan.ensure_state()
    out: dict = {"stress": {}, "profiles": {}}
    available = set((getattr(st, "strain_raw", {}) or {}).keys())
    for label in opts.labels:
        if label not in available:
            continue
        if opts.do_stress:
            res = E.compute_stress(scan, label=label, mode=opts.stress_mode, log=log)
            if res is not None:
                out["stress"][label] = res
    if opts.lines:
        try:
            out["profiles"] = E.extract_line_profiles(
                scan, opts.lines, line_width=opts.line_width)
        except Exception as exc:
            _log(log, f"[{scan.name}] line profiles skipped: {exc}")
    return out


def analyze_all(scans: list[E.Scan], opts: AnalyzeOptions | None = None, *,
                log: Log = None, progress: Progress = None) -> dict:
    """Light ANALYSIS over N scans. Returns {scan_name: analyze_scan(...)}."""
    opts = opts or AnalyzeOptions()
    out: dict = {}
    n = len(scans)
    for i, scan in enumerate(scans):
        _emit(progress, ProgressEvent(i, n, scan.name, "", "analyze", "stress + profiles"))
        out[scan.name] = analyze_scan(scan, opts, log=log)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hydrate saved workspaces (for an analysis-only session — Phase-1 loader)
# ─────────────────────────────────────────────────────────────────────────────

def hydrate_from_dirs(dirs: list[str], *, log: Log = None,
                      workspace_root: str | None = None) -> list[E.Scan]:
    """Rebuild Scan objects from saved workspace directories (no recompute).

    Accepts either a scan's results dir or its inner ``data`` dir. Failed loads
    are logged and skipped so one corrupt workspace doesn't sink the rest.

    When ``workspace_root`` is set (or inferable from *dirs*), looks for
    ``Parametros_cal.json`` / ``fast4d_session.json`` beside the batch and applies
    raw paths, braggpeaks, calibration ``h5_path`` (``<stem>.h5``, not braggpeaks)
    and calibration params.
    """
    scans: list[E.Scan] = []
    for d in dirs:
        dd = Path(d)
        data_dir = dd / "data" if (dd / "data").is_dir() else dd
        name = dd.parent.name if dd.name == "data" else dd.name
        scan = E.Scan(name=name)
        try:
            E.load_results(scan, str(data_dir), log=log)
            scans.append(scan)
        except Exception as exc:
            _log(log, f"Could not load workspace '{d}': {exc}")
    if scans and workspace_root:
        jp = E.find_workspace_params_json(workspace_root)
        if jp:
            E.apply_workspace_params_json(scans, jp, log=log)
    return scans
