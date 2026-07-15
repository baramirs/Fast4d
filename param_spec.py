"""fast4d.param_spec — framework-neutral calibration-parameter specification.

This is the single source of truth for *which* parameter belongs to *which*
step, its type, label and display rules. It has **no GUI dependency** (no Tk, no
Qt) so it can back the Qt param table, the icon strip, tests, or a script.

``CalibrationParams`` (in ``engine``) holds the values; this module describes how
to present and parse them. Extracted verbatim from the validated Tk param table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import engine as E


@dataclass(frozen=True)
class ParamSpec:
    field: str           # CalibrationParams attribute name
    label: str
    kind: str            # float|int|bool|enum|list2|list3|list4|points|fitted|scan_path
    options: tuple = ()  # for enum
    decimals: int = 4    # float display precision
    readonly: bool = False
    xy_display: bool = False  # list2 stored (y, x) natively; show/edit as (x, y)


# step_key → ordered parameter specs (pipeline order). step_key matches the icon
# keys used by the top strip. DETECTION steps (probe/select6/detection) are Path B.
PARAM_SPEC: dict[str, list[ParamSpec]] = {
    "probe": [
        ParamSpec("vacuum_path", "Vacuum file (per scan)", "scan_path", readonly=True),
        ParamSpec("probe_source", "Probe source", "enum",
                  ("vacuum", "bf_roi", "synthetic", "mean_dp")),
    ],
    "select6": [
        ParamSpec("six_points", "6 points on ADF (rx,ry)", "points", readonly=True),
    ],
    "detection": [
        ParamSpec("detect_min_absolute_intensity", "minAbsoluteIntensity", "int"),
        ParamSpec("detect_min_relative_intensity", "minRelativeIntensity", "float", decimals=3),
        ParamSpec("detect_min_peak_spacing", "minPeakSpacing", "int"),
        ParamSpec("detect_edge_boundary", "edgeBoundary", "int"),
        ParamSpec("detect_sigma", "sigma", "float", decimals=2),
        ParamSpec("detect_max_num_peaks", "maxNumPeaks", "int"),
        ParamSpec("detect_subpixel", "subpixel", "enum", ("none", "poly", "com")),
        ParamSpec("detect_corr_power", "corrPower", "float", decimals=2),
        ParamSpec("detect_cuda", "CUDA (GPU)", "bool"),
    ],
    "roi": [
        ParamSpec("roi_bounds", "ROI  [x0, x1, y0, y1]", "list4"),
    ],
    "origin": [
        ParamSpec("center_guess", "Center guess (x, y)", "list2", xy_display=True),
        ParamSpec("origin_sampling", "BVM sampling", "int"),
    ],
    "ellipse": [
        ParamSpec("ellipse_enabled", "Enabled", "bool"),
        ParamSpec("ellipse_center", "Ellipse center (x, y)", "list2", xy_display=True),
        ParamSpec("ellipse_q_range", "q_range (r0, r1)", "list2"),
        ParamSpec("ellipse_sampling", "Sampling", "int"),
        ParamSpec("ellipse_use_roi", "Use ROI", "bool"),
    ],
    "qpixel": [
        ParamSpec("cal_crystal", "Crystal", "enum", ("Si", "Au", "Custom")),
        ParamSpec("q_refit", "Refit px (off = use guess)", "bool"),
        ParamSpec("q_px", "px_guess (A^-1/px)", "float", decimals=5),
        ParamSpec("q_px_fitted", "px_fitted (A^-1/px)", "fitted", decimals=5, readonly=True),
        ParamSpec("q_kmax", "k_max", "float", decimals=2),
        ParamSpec("q_kpow", "bragg_k_power", "float", decimals=2),
        ParamSpec("q_use_roi", "Use ROI", "bool"),
    ],
    "basis": [
        ParamSpec("qr_rotation", "QR rotation (deg)", "float", decimals=3),
        ParamSpec("qr_flip", "QR flip", "bool"),
        ParamSpec("basis_manual_enabled", "Manual indices", "bool"),
        ParamSpec("index_origin", "index_origin", "int"),
        ParamSpec("index_g1", "index_g1", "int"),
        ParamSpec("index_g2", "index_g2", "int"),
        ParamSpec("min_spacing", "minSpacing", "int"),
        ParamSpec("min_absolute_intensity", "minAbsoluteIntensity", "int"),
        ParamSpec("max_num_peaks", "maxNumPeaks", "int"),
        ParamSpec("edge_boundary", "edgeBoundary", "int"),
        ParamSpec("vis_vmin", "vis vmin", "float", decimals=3),
        ParamSpec("vis_vmax", "vis vmax", "float", decimals=3),
        ParamSpec("zone_axis", "Zone axis [uvw]", "list3"),
        ParamSpec("real_axis_h", "Real axis H (+ry)", "list3"),
        ParamSpec("real_axis_v", "Real axis V (+rx)", "list3"),
        ParamSpec("indexing_seed", "Indexing RANSAC seed", "int"),
    ],
    "strain": [
        ParamSpec("coordinate_rotation", "coordinate_rotation (deg)", "float", decimals=2),
        ParamSpec("max_peak_spacing", "max_peak_spacing", "float", decimals=2),
        ParamSpec("vrange", "vrange (min, max) %", "list2"),
        ParamSpec("vrange_theta", "vrange_theta (min, max) deg", "list2"),
        ParamSpec("strain_scan_roi_bounds", "strain ROI [x0,x1,y0,y1]", "list4"),
        ParamSpec("strain_cmap", "Colormap (exx,eyy,exy)", "enum",
                  ("RdBu_r", "coolwarm", "seismic", "PuOr", "bwr")),
        ParamSpec("strain_cmap_theta", "Colormap (theta)", "enum",
                  ("PRGn", "twilight", "hsv", "PiYG")),
        ParamSpec("strain_layout", "Layout", "enum",
                  ("horizontal", "vertical", "square")),
        ParamSpec("strain_show_orientation", "Show orientation (theta)", "bool"),
    ],
    "tools": [
        ParamSpec("stress_material", "Stress material", "enum",
                  tuple(E.STRESS_MATERIALS.keys()) + ("Custom",)),
        ParamSpec("stress_symmetry", "Symmetry", "enum", ("cubic", "isotropic")),
        ParamSpec("stress_units", "Stress units", "enum", ("GPa", "MPa")),
        ParamSpec("stress_vmax", "Range ±vmax (0=auto)", "float", decimals=3),
    ],
}

# Human-friendly step titles.
STEP_TITLES = {
    "probe": "Probe", "select6": "6 Bragg points", "detection": "Detection",
    "roi": "ROI", "origin": "Origin", "ellipse": "Ellipse",
    "qpixel": "Q Pixel", "basis": "Basis", "strain": "Strain",
    "tools": "Stress / Tools",
}

# Tab grouping for the Excel-like param notebook (Qt). Detection's three Path-B
# steps share one tab; every calibration + ROI gets its own tab (user request).
#   (group_key, tab_title, [step_keys])
PARAM_GROUPS: list[tuple[str, str, list[str]]] = [
    ("detection", "Detection", ["probe", "select6", "detection"]),
    ("roi",       "ROI",       ["roi"]),
    ("origin",    "Origin",    ["origin"]),
    ("ellipse",   "Ellipse",   ["ellipse"]),
    ("qpixel",    "Q Pixel",   ["qpixel"]),
    ("basis",     "Basis",     ["basis"]),
    ("strain",    "Strain",    ["strain"]),
    ("tools",     "Stress",    ["tools"]),
]


def group_specs(group_key: str) -> list[tuple[str, ParamSpec]]:
    """Flattened [(step_key, spec), …] for a group's tab (preserves order)."""
    for gk, _title, steps in PARAM_GROUPS:
        if gk == group_key:
            out: list[tuple[str, ParamSpec]] = []
            for sk in steps:
                for spec in PARAM_SPEC.get(sk, []):
                    out.append((sk, spec))
            return out
    return []


def group_for_step(step_key: str) -> str | None:
    """Which tab group contains a given step key."""
    for gk, _title, steps in PARAM_GROUPS:
        if step_key in steps:
            return gk
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pure value <-> string conversion (no GUI; unit-tested)
# ─────────────────────────────────────────────────────────────────────────────

def format_value(spec: ParamSpec, value: Any) -> str:
    """CalibrationParams value → display string for a cell."""
    if value is None:
        return "—" if spec.kind == "fitted" else ""
    k = spec.kind
    if k in ("float", "fitted"):
        try:
            return f"{float(value):.{spec.decimals}f}"
        except (TypeError, ValueError):
            return str(value)
    if k == "int":
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)
    if k in ("list2", "list3", "list4"):
        if not value:
            return ""
        vals = list(value)
        # xy_display fields are stored (y, x) natively [pipeline + template convention]
        # but the user reads/edits them as (x, y) — swap only for display.
        if spec.xy_display and len(vals) == 2:
            vals = [vals[1], vals[0]]
        try:
            return ", ".join(f"{float(v):g}" for v in vals)
        except (TypeError, ValueError):
            return ", ".join(str(v) for v in vals)
    if k == "points":
        if not value:
            return "(none — pick 6 on ADF)"
        try:
            return " ".join(f"({float(x):.0f},{float(y):.0f})" for x, y in value)
        except (TypeError, ValueError):
            return str(value)
    if k == "scan_path":
        if not value:
            return "(not set — Choose vacuum file…)"
        from pathlib import Path
        return Path(str(value)).name
    return str(value)


def parse_value(spec: ParamSpec, text: str) -> Any:
    """Display string → CalibrationParams value. Raises ValueError on bad input."""
    k = spec.kind
    s = (text or "").strip()
    if k == "float":
        return float(s)
    if k == "int":
        return int(float(s))   # tolerate "2.0"
    if k in ("list2", "list3", "list4"):
        if not s:
            return [] if k != "list2" else [0.0, 0.0]
        parts = [p for p in s.replace(";", ",").split(",") if p.strip() != ""]
        nums = [float(p) for p in parts]
        n_expected = {"list2": 2, "list3": 3, "list4": 4}[k]
        if len(nums) != n_expected:
            raise ValueError(f"expected {n_expected} numbers, got {len(nums)}")
        # the user types xy_display fields as (x, y); store them back as (y, x)
        if spec.xy_display and len(nums) == 2:
            nums = [nums[1], nums[0]]
        if k == "list3":
            return [int(round(v)) for v in nums]
        return nums
    return s   # enum/bool handled by widgets
