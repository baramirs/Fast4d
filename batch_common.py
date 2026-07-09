"""Shared types and constants for multi-scan batch workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

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
