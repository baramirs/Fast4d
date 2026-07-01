"""Shared types and constants for multi-scan batch workflow."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from state import WorkflowState

# How a calibration step is applied across the batch list (after «Test over all» preview).
APPLY_MODE_ALL_SAME = "all_same"
APPLY_MODE_ONE_BY_ONE = "one_by_one"
# Legacy key kept so old sessions map to «same for all».
APPLY_MODE_ALL_THEN_EDIT = "all_then_edit"

APPLY_MODE_LABELS: dict[str, str] = {
    APPLY_MODE_ALL_SAME: "Same for all scans",
    APPLY_MODE_ONE_BY_ONE: "One scan at a time",
}

APPLY_MODE_VALUES = (APPLY_MODE_ALL_SAME, APPLY_MODE_ONE_BY_ONE)

# Per-scan export root next to raw .mib files (figures/ + data/ per scan stem).
BATCH_GUI_RESULTS_DIR = "gui_results_batch"

# Line-profile component order (maps sampled along each segment).
STRAIN_LINE_COMPONENTS: tuple[str, ...] = ("exx", "eyy", "exy")
STRESS_LINE_SIGMA_COMPONENTS: tuple[str, ...] = ("sigma_xx", "sigma_xy", "sigma_yy")


def scan_paths_match(a: Path | str | None, b: Path | str | None) -> bool:
    """True when two raw-scan paths refer to the same file (resolved comparison)."""
    if a is None or b is None:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


_DRIFT_ID_COLS = ("stem", "scan_id", "scan", "file", "name")
_DRIFT_DY_COLS = ("dy", "shift_dy_px", "shift_dy", "dy_px", "y_shift")
_DRIFT_DX_COLS = ("dx", "shift_dx_px", "shift_dx", "dx_px", "x_shift")


def _drift_row_value(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    # DictReader keys may include a UTF-8 BOM on the first column name
    for rk, val in row.items():
        if not rk:
            continue
        norm = rk.lstrip("\ufeff").strip().lower()
        if norm in keys and val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _drift_row_float(row: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return float(raw)
        except (ValueError, TypeError):
            continue
    for rk, raw in row.items():
        if not rk:
            continue
        norm = rk.lstrip("\ufeff").strip().lower()
        if norm not in keys or raw is None or str(raw).strip() == "":
            continue
        try:
            return float(raw)
        except (ValueError, TypeError):
            continue
    return None


def parse_drift_shifts_csv_rows(
    rows: list[dict],
) -> dict[str, tuple[float, float]]:
    """Parse drift CSV rows into ``{id: (dy, dx)}`` in pixels.

    Supports legacy ``stem, dy, dx`` and repro4dstem
    ``registration_shifts.csv`` (``scan_id, shift_dy_px, shift_dx_px``).
    """
    offsets: dict[str, tuple[float, float]] = {}
    for row in rows:
        scan_key = _drift_row_value(row, _DRIFT_ID_COLS)
        if not scan_key:
            continue
        dy = _drift_row_float(row, _DRIFT_DY_COLS)
        dx = _drift_row_float(row, _DRIFT_DX_COLS)
        if dy is None:
            dy = 0.0
        if dx is None:
            dx = 0.0
        offsets[scan_key] = (dy, dx)
    return offsets


def resolve_drift_shift(
    stem: str,
    shifts: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Look up ``(dy, dx)`` for a batch scan stem (exact or fuzzy ``scan_id`` match)."""
    if not stem or not shifts:
        return (0.0, 0.0)
    if stem in shifts:
        return shifts[stem]
    stem_l = stem.lower()
    for key, val in shifts.items():
        if key.lower() == stem_l:
            return val
    best: tuple[float, float] | None = None
    best_len = -1
    for key, val in shifts.items():
        kl = key.lower()
        if stem_l in kl or kl in stem_l:
            if len(key) > best_len:
                best_len = len(key)
                best = val
    return best if best is not None else (0.0, 0.0)


def apply_drift_to_line_segment(
    p0: tuple[float, float] | Any,
    p1: tuple[float, float] | Any,
    dy: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Move line segment opposite to this file's drift — **Y only** (X stays 0→511)."""
    return (
        (float(p0[0]), float(p0[1]) - dy),
        (float(p1[0]), float(p1[1]) - dy),
    )


def apply_drift_to_roi_bounds(
    bounds: tuple[int, int, int, int] | tuple[float, ...],
    dy: float,
    dx: float,
) -> tuple[int, int, int, int]:
    """Move ROI ``(x0, x1, y0, y1)`` opposite to this file's drift (X and Y)."""
    x0, x1, y0, y1 = bounds
    return (
        int(round(float(x0) - dx)),
        int(round(float(x1) - dx)),
        int(round(float(y0) - dy)),
        int(round(float(y1) - dy)),
    )


def drift_shift_key_matched(stem: str, shifts: dict[str, tuple[float, float]]) -> bool:
    """True if *shifts* contains a row for this batch scan stem (exact or fuzzy)."""
    if not stem or not shifts:
        return False
    if stem in shifts:
        return True
    stem_l = stem.lower()
    for key in shifts:
        kl = key.lower()
        if stem_l == kl or stem_l in kl or kl in stem_l:
            return True
    return False


@dataclass
class BatchScanItem:
    raw_path: Path
    h5_path: Path
    adf_thumb: Any = None
    detect_params_override: dict | None = None
    status: str = "pending"
    braggpeaks_path: Path | None = None
    compute_started_at: float | None = None
    compute_elapsed_s: float | None = None
    # Per-scan calibration overrides (None → use batch template for that step).
    roi_bounds: tuple[int, int, int, int] | None = None  # x0, x1, y0, y1 on R-space ADF/BF
    center_guess: tuple[int, int] | None = None
    origin_done: bool = False
    ellipse_done: bool = False
    q_pixel_done: bool = False
    basis_done: bool = False
    strain_done: bool = False
    step_overrides: dict[str, Any] = field(default_factory=dict)
    params_path: Path | None = None  # path to params.json for this scan
    analysis_enabled: bool = True    # when False, Calculate All skips this scan

    def default_braggpeaks_path(self) -> Path:
        return self.raw_path.with_name(f"{self.raw_path.stem}braggpeaks.h5")

    def default_params_path(self) -> Path:
        return self.raw_path.parent / "gui_results_batch" / self.raw_path.stem / "data" / "params.json"


@dataclass
class BatchScanResult:
    """In-memory cache of computed arrays for one batch scan.  No disk I/O required.

    Populated by :func:`extract_scan_result` after the batch calculate-all loop
    completes for each scan.  Plugins operate exclusively on these cached arrays
    so the expensive braggpeaks / strain pipeline never needs to re-run.
    """

    stem: str
    raw_path: Path
    image_shape: tuple[int, int] = (0, 0)         # (H, W) real-space scan grid
    r_pixel_nm: float | None = None               # nm/px real-space calibration

    # ── Calibration metadata ────────────────────────────────────────────────
    origin_xy: tuple[float, float] | None = None  # (y, x) fitted origin
    q_pixel_size: float | None = None             # Å⁻¹/px
    g1_qxy: tuple[float, float] | None = None     # basis vector 1 (qx, qy)
    g2_qxy: tuple[float, float] | None = None     # basis vector 2 (qx, qy)

    # ── Image data ─────────────────────────────────────────────────────────
    adf_array: np.ndarray | None = None           # ADF real-space image (H, W)
    probe_array: np.ndarray | None = None         # Probe (or BF virtual image) for report

    # ── Strain outputs — mirror WorkflowState fields ───────────────────────
    strain_raw: dict = field(default_factory=dict)
    # {"without_roi": ndarray (H, W, 3), "with_roi": ndarray (H, W, 3)}
    # Channel order matches py4DSTEM: 0=εyy, 1=εxx, 2=εxy

    strain_arrays: dict = field(default_factory=dict)
    # {"without_roi": ndarray (H, W), "with_roi": ndarray (H, W)}
    # Primary 2-D array (εxx in percent) for quick display / analysis

    # ── Stress outputs ─────────────────────────────────────────────────────
    stress_tensors_pa: dict = field(default_factory=dict)
    # {label: {"sigma_xx": ndarray, "sigma_yy": ndarray, "sigma_xy": ndarray}}
    # Cauchy stress in Pa

    # ── Line profile positions ─────────────────────────────────────────────
    line_profiles_px: dict = field(default_factory=dict)
    # {lid: ((x0, y0), (x1, y1))} — pixel coordinates shared across all scans
    # Populated from state.fixed_line_profiles_px before calculate-all runs
    line_profile_width: int = 3             # integration width used when computing

    # ── Calibration overview figure ────────────────────────────────────────
    cal_figure: Any = None
    # matplotlib.figure.Figure showing calibrated diffraction space:
    # 2-D peak histogram + origin marker + g1/g2 basis arrows.
    # Generated in _batch_calculate_all after _batch_prepare_for_strain.

    # ── py4DSTEM-generated figures (preserved from compute steps) ──────────
    strain_figures: dict = field(default_factory=dict)
    # {label: matplotlib.figure.Figure}  — the real 4-panel strain figure
    # produced by compute_strain_map_step (has proper colorbars, vrange, etc.)

    stress_figures: dict = field(default_factory=dict)
    # {label: matplotlib.figure.Figure}  — stress figure from compute_stress_analysis_step

    basis_preview_figures: list = field(default_factory=list)
    # choose_basis_vectors panel(s) from Step 12 preview (one or more matplotlib figures)

    line_figure: Any = None
    # Combined εxx-map + multi-line-profile figure from compute_multi_line_profiles_figure.

    stress_line_profile_data: dict = field(default_factory=dict)
    stress_meta: dict = field(default_factory=dict)

    # ── Calstate summary ───────────────────────────────────────────────────
    calstate: dict = field(default_factory=dict)
    # Same format as single_scan_cal_ui_flags():
    # {"origin": "applied"|"staged"|"pending",
    #  "ellipse": "applied"|"staged"|"pending"|"unused",
    #  "qpx": ..., "basis": ..., "strain": ..., "stress": ..., "lines": ...}


def extract_scan_result(state: "WorkflowState", item: BatchScanItem) -> "BatchScanResult":
    """Snapshot calibration + computed arrays from *state* into a :class:`BatchScanResult`.

    Called immediately after the per-scan calculate-all pipeline finishes,
    while the WorkflowState still holds that scan's data.
    """
    try:
        from pipeline import single_scan_cal_ui_flags
    except ImportError:
        from .pipeline import single_scan_cal_ui_flags

    result = BatchScanResult(
        stem=item.raw_path.stem,
        raw_path=item.raw_path,
    )

    # ── Shape / calibration ─────────────────────────────────────────────
    dc = getattr(state, "datacube", None)
    if dc is not None:
        try:
            rshape = getattr(dc, "Rshape", None)
            if rshape is not None and len(rshape) >= 2:
                result.image_shape = (int(rshape[0]), int(rshape[1]))
        except Exception:
            pass

    result.r_pixel_nm = getattr(state, "image_pixel_size", None)
    result.q_pixel_size = getattr(state, "q_pixel_size", None)

    # Basis vectors
    sbp = getattr(state, "strain_basis_params", {}) or {}
    result.g1_qxy = sbp.get("g1_qxy")
    result.g2_qxy = sbp.get("g2_qxy")

    # Origin
    og = getattr(state, "origin_fit", None) or getattr(state, "center_guess", None)
    if og is not None:
        try:
            result.origin_xy = (float(og[0]), float(og[1]))
        except Exception:
            pass

    # ── ADF thumbnail ───────────────────────────────────────────────────
    vi = getattr(state, "virtual_images", None) or {}
    adf = vi.get("annular_dark_field") or item.adf_thumb
    if adf is not None:
        try:
            result.adf_array = np.asarray(adf)
        except Exception:
            pass

    # ── Probe (Fourier-space) ────────────────────────────────────────
    probe = getattr(state, "probe", None)
    if probe is not None:
        try:
            pa = np.asarray(probe)
            result.probe_array = pa
        except Exception:
            pass

    # ── Strain ─────────────────────────────────────────────────────────
    sr = getattr(state, "strain_raw", None) or {}
    for label, arr in sr.items():
        if arr is None:
            continue
        try:
            if isinstance(arr, dict):
                result.strain_raw[label] = dict(arr)
            else:
                result.strain_raw[label] = np.asarray(arr, dtype=float)
        except Exception:
            pass

    sa = getattr(state, "strain_arrays", None) or {}
    for label, arr in sa.items():
        if arr is not None:
            try:
                result.strain_arrays[label] = np.asarray(arr, dtype=float)
            except Exception:
                pass

    # ── Stress ─────────────────────────────────────────────────────────
    stp = getattr(state, "stress_tensors_pa", None) or {}
    for label, sigma_dict in stp.items():
        try:
            result.stress_tensors_pa[label] = {
                k: np.asarray(v, dtype=float)
                for k, v in sigma_dict.items()
                if v is not None
            }
        except Exception:
            pass

    # ── Line profile positions (global fixed segments) ──────────────────
    fp = getattr(state, "fixed_line_profiles_px", None) or {}
    if fp:
        result.line_profiles_px = dict(fp)

    slp = getattr(state, "stress_line_profile_data", None) or {}
    if slp:
        result.stress_line_profile_data = {
            k: dict(v) if isinstance(v, dict) else v for k, v in slp.items()
        }
    sm = getattr(state, "stress_meta", None) or {}
    if sm:
        result.stress_meta = {k: dict(v) if isinstance(v, dict) else v for k, v in sm.items()}

    bpf = getattr(state, "basis_preview_figures", None) or []
    if bpf:
        result.basis_preview_figures = [
            f for f in bpf if f is not None and hasattr(f, "savefig")
        ]

    # ── Calstate ────────────────────────────────────────────────────────
    try:
        result.calstate = single_scan_cal_ui_flags(state)
    except Exception:
        pass

    return result


def batch_scan_cal_flags(result: "BatchScanResult") -> dict[str, str]:
    """Return calstate dict from a cached :class:`BatchScanResult`.

    Falls back to a minimal reconstruction if ``result.calstate`` is empty
    (e.g. when only partial computation was done).
    """
    if result.calstate:
        return dict(result.calstate)

    # Minimal reconstruction from cached arrays
    flags: dict[str, str] = {}
    flags["origin"] = "applied" if result.origin_xy is not None else "pending"
    flags["ellipse"] = "unused"   # batch doesn't track ellipse outcome here
    flags["qpx"] = "applied" if result.q_pixel_size else "pending"
    flags["basis"] = "applied" if result.g1_qxy is not None else "pending"
    flags["strain"] = "applied" if result.strain_raw else "pending"
    flags["stress"] = "applied" if result.stress_tensors_pa else "pending"
    flags["lines"] = "applied" if result.line_profiles_px else "pending"
    return flags


def batch_item_cal_ui_flags(
    item: "BatchScanItem",
    *,
    result: "BatchScanResult | None" = None,
    workflow_state: "WorkflowState | None" = None,
) -> dict[str, str]:
    """Best-effort calibration-state map for a batch scan, in the same shape
    as :func:`single_scan_cal_ui_flags`.

    Resolution order for each step:
      1. If ``workflow_state.braggpeaks`` matches this item (loaded for compute),
         delegate to :func:`single_scan_cal_ui_flags` so the bottom-bar lights
         reflect the live truth (origin written, calstate flags, etc.).
      2. Otherwise fall back to the cached :class:`BatchScanResult` and the
         item's persistent flags (``origin_done``, ``q_pixel_done`` …) and
         pending overrides (``step_overrides``) — so we can still answer
         "staged" vs "pending" before the scan has been loaded.

    Keys: origin, ellipse, qpx, basis, strain, stress, lines.
    Values: ``applied`` | ``staged`` | ``pending`` | ``unused``.
    """
    # ── 1) Live state available → use the single-scan reader directly ──
    if workflow_state is not None and getattr(workflow_state, "braggpeaks", None) is not None:
        try:
            try:
                from pipeline import single_scan_cal_ui_flags
            except ImportError:
                from .pipeline import single_scan_cal_ui_flags  # type: ignore
            return single_scan_cal_ui_flags(workflow_state)
        except Exception:
            pass

    # ── 2) Reconstruct from item flags + overrides + cached result ──
    flags: dict[str, str] = {}
    overrides = dict(getattr(item, "step_overrides", {}) or {})

    if bool(item.origin_done):
        flags["origin"] = "applied"
    elif item.center_guess is not None or overrides.get("origin"):
        flags["origin"] = "staged"
    else:
        flags["origin"] = "pending"

    ell_st = overrides.get("step10")
    if bool(item.ellipse_done):
        flags["ellipse"] = "applied"
    elif isinstance(ell_st, dict) and ell_st:
        flags["ellipse"] = "staged"
    else:
        flags["ellipse"] = "unused"

    if bool(item.q_pixel_done):
        flags["qpx"] = "applied"
    elif (overrides.get("step11") or {}).get("px_guess"):
        flags["qpx"] = "staged"
    else:
        flags["qpx"] = "pending"

    if bool(item.basis_done):
        flags["basis"] = "applied"
    elif isinstance(overrides.get("basis"), dict) and overrides["basis"]:
        flags["basis"] = "staged"
    else:
        flags["basis"] = "pending"

    if result is not None:
        if result.strain_raw:
            flags["strain"] = "applied"
        else:
            flags["strain"] = "staged" if item.strain_done else "pending"
        flags["stress"] = "applied" if result.stress_tensors_pa else "pending"
        flags["lines"] = "applied" if result.line_profiles_px else "pending"
    else:
        flags["strain"] = "applied" if bool(item.strain_done) else "pending"
        flags["stress"] = "pending"
        flags["lines"] = "pending"

    return flags


# Calibration steps after optional Bragg compute (step_id → menu label).
BATCH_CALIBRATION_STEPS: tuple[tuple[str, str], ...] = (
    ("step5_roi", "Step 1 — Select ROI"),
    ("step9", "Step 6 — Origin correction"),
    ("step10", "Step 7 — Ellipse calibration (Optional)"),
    ("step11", "Step 8 — Q pixel size"),
    ("step12", "Step 9 — Strain basis"),
    # step13 / step14 are now unified into the "Step 10" tools section appended
    # at the bottom of the calibration scroll — not standalone accordion items.
)

# Short tab titles in the calibration notebook.
BATCH_CALIBRATION_TAB_LABELS: dict[str, str] = {
    "step5_roi": "Step 1 - ROI",
    "step9": "Step 6 - Origin",
    "step10": "Step 7 - Ellipse",
    "step11": "Step 8 - Q pixel",
    "step12": "Step 9 - Strain basis",
}

# Figure tab / suptitle names (no "Step N" prefix).
BATCH_CALIBRATION_DISPLAY_NAMES: dict[str, str] = {
    "step5_roi": "ROI",
    "step5": "ROI",
    "step5_test": "ROI (preview)",
    "step9": "Origin correction",
    "step9_test": "Origin correction (preview)",
    "step10": "Ellipse calibration",
    "step11": "Q pixel size",
    "step11_test": "Q pixel size (preview)",
    "step12": "Strain basis",
    "step12_test": "Strain basis (preview)",
    "step13": "Strain map",
    "step14": "Strain ROI",
    "bragg": "Bragg peaks",
    "detect": "Disk detection",
    "load": "Loading data",
    "probe": "Shared probe",
}


def batch_calibration_display_name(step_id: str) -> str:
    """Human-readable calibration name for activity lines and figure titles."""
    sid = (step_id or "").strip()
    if sid in BATCH_CALIBRATION_DISPLAY_NAMES:
        return BATCH_CALIBRATION_DISPLAY_NAMES[sid]
    for key in sorted(BATCH_CALIBRATION_DISPLAY_NAMES, key=len, reverse=True):
        if sid.startswith(key):
            return BATCH_CALIBRATION_DISPLAY_NAMES[key]
    for key, tab_label in BATCH_CALIBRATION_TAB_LABELS.items():
        if sid.startswith(key):
            if " - " in tab_label:
                return tab_label.split(" - ", 1)[1].strip()
            return tab_label.replace("Step ", "").strip()
    return sid.replace("_", " ")


def batch_figure_title(step_id: str, scan_stem: str) -> str:
    """Tab title and matplotlib suptitle: «{calibration} — {scan}»."""
    stem = str(scan_stem).strip() or "scan"
    return f"{batch_calibration_display_name(step_id)} — {stem}"

# Treeview columns (batch status table).
BATCH_TABLE_COLS: tuple[str, ...] = (
    "name",
    "analysis",  # ✓/✗ — whether Calculate All will include this scan
    "bragg",
    "roi",
    "origin",
    "ellipse",
    "q_pixel",
    "basis",
    "strain",    # ✓/◐/✗ after calculate-all
    "stress",    # ✓/◐/✗ after calculate-all
    "lines",     # ✓/◐/✗ after calculate-all
    "status",
    "time",
)

# Default split in the right column: fraction of height for figures (rest = table).
BATCH_DEFAULT_FIGURES_HEIGHT_RATIO = 0.58

# Inline ADF strip at the bottom of the batch window (canvas + label frame chrome).
BATCH_ADF_GALLERY_CANVAS_HEIGHT = 112
BATCH_ADF_BOTTOM_EXTRA_PX = 40
