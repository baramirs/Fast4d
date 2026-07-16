"""fast4d.engine — UI-free 4D-STEM strain analysis engine.

This is the orchestration layer for the unified Fast4D GUI. It is intentionally
free of any Tkinter / widget code so it can be driven by the GUI, the multi-file
driver, a script, or a test.

It wraps the proven, battle-tested pieces of the existing codebase:
    * ``pipeline.py``        — py4DSTEM calibration + strain step functions
    * ``fast_artifacts.py``  — save/load workspace (npz + manifest)
    * ``fast_batch.py``      — per-scan calibration application + line profiles
    * ``stress_analysis.py`` — Hooke's-law stress (via pipeline.compute_stress_analysis_step)

and grounds the sequence + default parameters in the reference notebook
``4Dstrain-analysis.ipynb``.

──────────────────────────────────────────────────────────────────────────────
COMPUTE vs ANALYSIS  (the core "fast" idea, made explicit in the notebook)
──────────────────────────────────────────────────────────────────────────────
COMPUTE  (heavy, run once, results persisted):
    load_braggpeaks → calibrate (roi/origin/ellipse/q-pixel/basis) → compute_strain
    → save_results
ANALYSIS (light, run many times, reads persisted strain arrays — never recomputes):
    load_results → compute_stress / extract_line_profiles

──────────────────────────────────────────────────────────────────────────────
SINGLE vs MULTI
──────────────────────────────────────────────────────────────────────────────
There is no "mode". A scan is a scan. The GUI populates the same calibration
fields whether there is 1 scan or N. With N scans, a *template* (a single
``CalibrationParams`` or ``Parametros_cal.json``) fills each scan's fields and
``compute_strain`` is run serially per scan. See ``driver.py``.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from _bootstrap import bootstrap_sys_path

bootstrap_sys_path()


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports of the proven engine pieces (kept lazy so importing this module
# never drags py4DSTEM in until a step actually runs).
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline():
    import pipeline
    return pipeline


def _artifacts():
    import fast_artifacts
    return fast_artifacts


def _batch():
    import fast_batch
    return fast_batch


def _new_state():
    from state import WorkflowState
    return WorkflowState()


Log = Callable[[str], None] | None


def _log(log: Log, msg: str) -> None:
    if log is not None:
        try:
            log(msg)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# TWO distinct material roles (a key domain point):
#
#   1. CAL CRYSTAL  — the reference crystal used to fit the Q-pixel size via
#                     structure factors (notebook cell 37). Often a calibration
#                     standard (Au), independent of the sample.
#   2. STRESS MATERIAL — the *sample's* elastic constants for Hooke's-law stress.
#                        How many independent constants there are depends on the
#                        crystal symmetry: cubic (Si, Au) → C11, C12, C44 (3);
#                        isotropic → 2 (here expressed as C11, C12 with
#                        C44 = (C11 - C12)/2).
#
# Atomic positions below are the fractional coordinates from notebook cell 37.
# ─────────────────────────────────────────────────────────────────────────────

# FCC (Au) — 4-atom basis
_FCC_POS = [
    [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
]
# Diamond cubic (Si) — 8-atom basis
_DIAMOND_POS = [
    [0.0, 0.0, 0.0],   [0.25, 0.25, 0.25],
    [0.0, 0.5, 0.5],   [0.25, 0.75, 0.75],
    [0.5, 0.0, 0.5],   [0.75, 0.25, 0.75],
    [0.5, 0.5, 0.0],   [0.75, 0.75, 0.25],
]


@dataclass(frozen=True)
class CalCrystal:
    """Reference crystal for Q-pixel calibration (structure factors)."""
    name: str
    a_lat: float          # cubic lattice constant (Å)
    atom_num: int         # representative Z
    positions: tuple      # fractional coordinates (N×3)


CAL_CRYSTALS: dict[str, CalCrystal] = {
    # Si first per user preference; Au is the notebook's calibration standard.
    "Si": CalCrystal("Si", 5.431, 14, tuple(map(tuple, _DIAMOND_POS))),
    "Au": CalCrystal("Au", 4.078, 79, tuple(map(tuple, _FCC_POS))),
}
DEFAULT_CAL_CRYSTAL = "Si"


# ── Custom crystal builder: pick a structure type → generate the positions array ──
# (so the user edits element(s) + lattice parameter + structure, NOT raw coordinates).
ELEMENT_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Sr": 38, "Y": 39,
    "Zr": 40, "Nb": 41, "Mo": 42, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Cs": 55, "Ba": 56, "Ta": 73, "W": 74, "Pt": 78,
    "Au": 79, "Pb": 82,
}

# fractional positions (conventional cubic cell) for single-species structures
_STRUCT_POS = {
    "simple_cubic": [[0.0, 0.0, 0.0]],
    "bcc": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    "fcc": list(map(list, _FCC_POS)),
    "diamond": list(map(list, _DIAMOND_POS)),
}
# two-species structures: (sublattice A positions, sublattice B positions)
_QTR = [[x + 0.25, y + 0.25, z + 0.25] for (x, y, z) in _FCC_POS]
_HALF_X = [[x + 0.5, y, z] for (x, y, z) in _FCC_POS]
_STRUCT_POS2 = {
    "zincblende": (list(map(list, _FCC_POS)), _QTR),       # e.g. GaAs (diamond, 2 species)
    "rocksalt": (list(map(list, _FCC_POS)), _HALF_X),      # e.g. NaCl
    "cscl": ([[0.0, 0.0, 0.0]], [[0.5, 0.5, 0.5]]),        # e.g. CsCl
}
CRYSTAL_STRUCTURES = ("simple_cubic", "bcc", "fcc", "diamond", "zincblende", "rocksalt", "cscl")
TWO_SPECIES_STRUCTURES = set(_STRUCT_POS2)


def element_to_z(symbol_or_z) -> int:
    """'Si' / 'si' / 14 / '14' → atomic number (int). Raises on unknown symbol."""
    s = str(symbol_or_z).strip()
    if not s:
        raise ValueError("empty element")
    if s.isdigit():
        return int(s)
    z = ELEMENT_Z.get(s.capitalize())
    if z is None:
        raise ValueError(f"unknown element '{s}' (use a symbol like Si/Ge/Au or a Z number)")
    return int(z)


def build_custom_crystal(structure: str, a_lat: float, element_a, element_b=None) -> dict:
    """Generate a custom calibration crystal dict {a_lat, atom_num, positions}.

    ``structure`` ∈ CRYSTAL_STRUCTURES. Single-species (simple_cubic/bcc/fcc/diamond)
    use ``element_a`` for every site; two-species (zincblende/rocksalt/cscl) put
    ``element_a`` on sublattice A and ``element_b`` on sublattice B. ``element_*`` may
    be a symbol ('Si','Ge','Au') or a Z number. The positions array is GENERATED — the
    user never hand-writes coordinates."""
    s = str(structure).strip().lower()
    za = element_to_z(element_a)
    if s in TWO_SPECIES_STRUCTURES:
        zb = element_to_z(element_b if element_b not in (None, "") else element_a)
        pa, pb = _STRUCT_POS2[s]
        positions = [list(map(float, p)) for p in (list(pa) + list(pb))]
        atom_num = [za] * len(pa) + [zb] * len(pb)
    elif s in _STRUCT_POS:
        positions = [list(map(float, p)) for p in _STRUCT_POS[s]]
        atom_num = za                              # single Z for all sites
    else:
        raise ValueError(f"unknown structure '{structure}' (use one of {CRYSTAL_STRUCTURES})")
    return {"a_lat": float(a_lat), "atom_num": atom_num, "positions": positions,
            "structure": s}


# ── CIF → CalCrystal / py4DSTEM Crystal (shared by Index BVM + Q-pixel) ───────

_CUBIC_LEN_RTOL = 1e-3
_CUBIC_ANG_ATOL_DEG = 0.5


def is_approximately_cubic(
    cell: tuple[float, float, float, float, float, float] | list[float] | np.ndarray,
    *,
    len_rtol: float = _CUBIC_LEN_RTOL,
    ang_atol_deg: float = _CUBIC_ANG_ATOL_DEG,
) -> bool:
    """True when a≈b≈c and α≈β≈γ≈90° (within tolerances)."""
    a, b, c, alpha, beta, gamma = (float(x) for x in np.asarray(cell, dtype=float).ravel()[:6])
    scale = max(abs(a), abs(b), abs(c), 1e-15)
    if max(abs(a - b), abs(b - c), abs(a - c)) > len_rtol * scale:
        return False
    return (
        abs(alpha - 90.0) <= ang_atol_deg
        and abs(beta - 90.0) <= ang_atol_deg
        and abs(gamma - 90.0) <= ang_atol_deg
    )


@dataclass(frozen=True)
class CifCrystalInfo:
    """Result of :func:`load_crystal_from_cif` (CalCrystal + cubic metadata)."""
    cal: CalCrystal
    cell: tuple[float, float, float, float, float, float]  # a,b,c,α,β,γ
    is_cubic: bool
    path: str
    warning: str | None = None


def load_crystal_from_cif(
    path: str | Path,
    *,
    primitive: bool = False,
    conventional_standard_structure: bool = True,
) -> CifCrystalInfo:
    """Load a CIF via py4DSTEM ``Crystal.from_CIF`` (needs pymatgen).

    For Index BVM v1 (cubic metric): if the cell is not cubic, ``cal.a_lat`` is
    still set to conventional ``a`` and ``warning`` explains the limitation.
    """
    from py4DSTEM.process.diffraction import Crystal

    cif = Path(path)
    if not cif.is_file():
        raise FileNotFoundError(f"CIF not found: {cif}")
    crystal = Crystal.from_CIF(
        str(cif),
        primitive=bool(primitive),
        conventional_standard_structure=bool(conventional_standard_structure),
    )
    cell_arr = np.asarray(crystal.cell, dtype=float).ravel()
    if cell_arr.size < 6:
        raise ValueError(f"CIF cell incomplete ({cell_arr.size} values): {cif}")
    cell = tuple(float(x) for x in cell_arr[:6])
    a, b, c, alpha, beta, gamma = cell
    cubic = is_approximately_cubic(cell)
    if cubic:
        a_lat = float(np.mean([a, b, c]))
        warning = None
    else:
        a_lat = float(a)
        warning = (
            f"CIF cell is not cubic "
            f"(a={a:.4f}, b={b:.4f}, c={c:.4f} Å; "
            f"α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}°). "
            f"Index BVM v1 uses effective a={a_lat:.4f} Å (cubic metric)."
        )
    positions = tuple(map(tuple, np.asarray(crystal.positions, dtype=float)))
    numbers = np.asarray(crystal.numbers).ravel().astype(int)
    if numbers.size == 0:
        raise ValueError(f"CIF has no atomic sites: {cif}")
    uniq = sorted({int(z) for z in numbers.tolist()})
    atom_num: int | list[int] = (
        [int(z) for z in numbers.tolist()] if len(uniq) > 1 else int(uniq[0])
    )
    cal = CalCrystal(cif.stem, a_lat, atom_num, positions)
    return CifCrystalInfo(
        cal=cal,
        cell=cell,
        is_cubic=cubic,
        path=str(cif.resolve()),
        warning=warning,
    )


@dataclass(frozen=True)
class StressMaterial:
    """Sample elastic constants for stress. ``symmetry`` sets how they reduce."""
    name: str
    symmetry: str         # "cubic" | "isotropic"
    c11_gpa: float
    c12_gpa: float
    c44_gpa: float        # for isotropic, treated as (C11 - C12)/2


STRESS_MATERIALS: dict[str, StressMaterial] = {
    # name           sym       C11     C12     C44
    "Si":  StressMaterial("Si",  "cubic", 165.7,  63.9,  79.6),
    "Au":  StressMaterial("Au",  "cubic", 192.9, 163.8,  41.5),
    "Ge":  StressMaterial("Ge",  "cubic", 129.0,  47.9,  67.1),
    "GaAs": StressMaterial("GaAs", "cubic", 118.8, 53.8,  59.4),
    "InAs": StressMaterial("InAs", "cubic",  83.3, 45.3,  39.6),
    "AlAs": StressMaterial("AlAs", "cubic", 120.2, 57.0,  58.9),
    "GaN":  StressMaterial("GaN",  "cubic", 390.0, 145.0, 105.0),
    "InP":  StressMaterial("InP",  "cubic", 102.2, 57.6,  46.0),
}
DEFAULT_STRESS_MATERIAL = "Si"   # user's sample material


# ─────────────────────────────────────────────────────────────────────────────
# Calibration parameters — one flat, GUI-friendly bundle per scan.
# Field names + defaults are grounded in the notebook cells noted in comments.
# This is what fills the calibration text boxes (single) or table columns (multi).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationParams:
    # ── DETECTION (Path B, OPTIONAL — only when no braggpeaks.h5 yet) ──────────
    # Probe/template comes from the vacuum; the 6 points are picked on the DATA's
    # ADF (real-space scan positions) to preview detection and tune detect_params.
    probe_source: str = "vacuum"     # "vacuum" | "bf_roi" | "synthetic" | "mean_dp"
    probe_vacuum_roi_bounds: list = field(default_factory=list)  # [x0,x1,y0,y1] on ADF (bf_roi)
    six_points: list = field(default_factory=list)   # 6 (rx, ry) on the ADF
    # find_Bragg_disks params (distinct from the basis params below)
    detect_min_absolute_intensity: int = 3
    detect_min_relative_intensity: float = 0.05
    detect_min_peak_spacing: int = 10
    detect_edge_boundary: int = 10
    detect_sigma: float = 0.0
    detect_max_num_peaks: int = 50
    detect_subpixel: str = "poly"    # "none" | "poly" | "com"
    detect_corr_power: float = 1.0
    detect_cuda: bool = True         # GPU — important for the heavy detection

    # ── ROI (cell 12) — calibration region for q-pixel / basis fits ───────────
    roi_bounds: list = field(default_factory=list)        # [x0, x1, y0, y1] or []

    # ── Origin (cells 23-25) — uses a center "pickup", NOT the ROI ────────────
    center_guess: list = field(default_factory=lambda: [128.0, 128.0])  # (y, x)
    origin_sampling: int = 2

    # ── Ellipse (cells 30-35) — OPTIONAL, OFF by default; uses the shared ROI ──
    ellipse_enabled: bool = False
    ellipse_q_range: list = field(default_factory=lambda: [40, 60])  # (r0, r1)
    ellipse_sampling: int = 1
    ellipse_use_roi: bool = True
    ellipse_center: list | None = None  # (y, x) BVM-px override for fit center;
                                         # None = use bvm.origin (legacy default)

    # ── Q pixel size (cell 37) — always fits: px_guess in → fitted px out ──────
    q_px: float = 0.0137          # px_guess (editable INPUT; never overwritten)
    q_px_fitted: float | None = None   # fitted OUTPUT (read-only; set on refit)
    q_refit: bool = True          # True (DEFAULT, user confirmed) → fit px from structure
                                  # factors (calibrate_pixel_size; the guess seeds it). The
                                  # q-pixel figure's blue box shows guess vs fitted + delta.
                                  # False → use the guess px directly (no fit).
    q_kmax: float = 1.0           # k_max (Å⁻¹)
    q_kpow: float = 1.0           # bragg_k_power
    q_use_roi: bool = False       # uses the shared ROI when True

    # ── Basis / QR (cell 39) ──────────────────────────────────────────────────
    qr_rotation: float = 0.0
    qr_flip: bool = False
    basis_manual_enabled: bool = False
    index_origin: int = 0
    index_g1: int = 3
    index_g2: int = 4
    min_spacing: int = 10
    min_absolute_intensity: int = 80
    max_num_peaks: int = 60
    edge_boundary: int = 4
    vis_vmin: float = 0.0
    vis_vmax: float = 0.995
    # BVM indexing (RANSAC + hkl / zone axis) — used by IndexerDialog before basis
    zone_axis: list = field(default_factory=lambda: [1, 1, 0])
    real_axis_h: list = field(default_factory=lambda: [0, 0, -1])   # +ry (left→right)
    real_axis_v: list = field(default_factory=lambda: [-1, 1, 0])  # +rx (top→bottom)
    indexing_tol_px: float | None = None   # None → use max_peak_spacing
    indexing_seed: int = 0
    # "unknown" = lattice + g1/g2 for Basis (no absolute hkl);
    # "known" = anchor with zone + real axes + QR
    indexing_orientation_mode: str = "unknown"

    # ── Strain mapping (cell 41) ──────────────────────────────────────────────
    coordinate_rotation: float = 0.0
    max_peak_spacing: float = 2.0
    vrange: list = field(default_factory=lambda: [-5.0, 5.0])
    vrange_theta: list = field(default_factory=lambda: [-5.0, 5.0])
    strain_scan_roi_bounds: list | None = None   # ROI for the "with ROI" pass
    strain_cmap: str = "RdBu_r"           # colormap for exx/eyy/exy (py4DSTEM show_strain)
    strain_cmap_theta: str = "PRGn"       # colormap for theta (py4DSTEM show_strain)
    strain_layout: str = "horizontal"     # "horizontal" | "vertical" | "square" (py4DSTEM show_strain)
    strain_show_orientation: bool = True  # if False, theta panel is hidden post-render

    # ── Q-pixel calibration crystal (structure factors) ──────────────────────
    # Shared with Index BVM (lattice_a). "CIF" uses cif_path via from_CIF.
    cal_crystal: str = DEFAULT_CAL_CRYSTAL           # "Si" | "Au" | "Custom" | "CIF"
    custom_crystal: dict | None = None               # {a_lat, atom_num, positions}
    cif_path: str | None = None                      # path to .cif when cal_crystal=="CIF"

    # ── Stress: sample material + symmetry (elastic constants) ────────────────
    stress_material: str = DEFAULT_STRESS_MATERIAL   # "Si" | … | "Custom"
    stress_symmetry: str = "cubic"                   # "cubic" | "isotropic"
    custom_stress: dict | None = None                # {c11_gpa, c12_gpa, c44_gpa}
    stress_units: str = "MPa"                        # stress-map display units: "GPa" | "MPa"
    stress_vmax: float = 0.0                          # symmetric colour range ±vmax (display units); 0 = auto

    # convenience -------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def cal_crystal_obj(self) -> CalCrystal:
        """Resolve the Q-pixel / Index BVM crystal (Si/Au/Custom/CIF)."""
        if self.cal_crystal == "CIF" and self.cif_path:
            return load_crystal_from_cif(self.cif_path).cal
        if self.cal_crystal == "Custom" and self.custom_crystal:
            c = self.custom_crystal
            an = c.get("atom_num", 14)
            atom_num = [int(z) for z in an] if isinstance(an, (list, tuple)) else int(an)
            return CalCrystal(
                "Custom",
                float(c.get("a_lat", 5.431)),
                atom_num,                          # int (single species) or list (per-site Z)
                tuple(map(tuple, c.get("positions", _DIAMOND_POS))),
            )
        return CAL_CRYSTALS.get(self.cal_crystal, CAL_CRYSTALS[DEFAULT_CAL_CRYSTAL])

    def stress_constants_gpa(self) -> tuple[float, float, float]:
        """Resolve (C11, C12, C44) in GPa for the chosen sample material/symmetry.

        For ``isotropic`` symmetry C44 is derived as (C11 - C12)/2.
        """
        if self.stress_material == "Custom" and self.custom_stress:
            c = self.custom_stress
            c11 = float(c.get("c11_gpa", 165.7))
            c12 = float(c.get("c12_gpa", 63.9))
            c44 = float(c.get("c44_gpa", 79.6))
        else:
            m = STRESS_MATERIALS.get(self.stress_material,
                                     STRESS_MATERIALS[DEFAULT_STRESS_MATERIAL])
            c11, c12, c44 = m.c11_gpa, m.c12_gpa, m.c44_gpa
        if self.stress_symmetry == "isotropic":
            c44 = (c11 - c12) / 2.0
        return c11, c12, c44


# ─────────────────────────────────────────────────────────────────────────────
# Scan — a thin handle around the proven WorkflowState + bookkeeping.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scan:
    name: str
    raw_path: str = ""
    h5_path: str = ""
    vacuum_path: str = ""          # vacuum scan → probe template (Path B detection)
    braggpeaks_path: str = ""
    params: CalibrationParams = field(default_factory=CalibrationParams)
    params_source: str = ""            # path of a params/session JSON applied (for the file check-icon)
    state: Any = None                  # WorkflowState (lazy)
    status: str = "pending"            # pending|computed|done|error
    results_dir: str = ""              # where save_results wrote (data dir parent)
    figures_dir: str = ""
    figures: dict = field(default_factory=dict)   # {step_name: matplotlib Figure}
    figure_spill: dict = field(default_factory=dict)  # {key: png_path} when spilled from RAM
    adf_cache: Any = None              # ADF preview (from the light .h5) — persists across
                                       # state resets (a heavy datacube load clears
                                       # state.virtual_images; this keeps pickers fed)
    lines: dict = field(default_factory=dict)   # {line_id: [[x0,y0],[x1,y1]]} placed on THIS scan
    # NOTE: area_roi / area_rois are ANALYSIS measurement ROIs (line-profile extraction),
    # NOT the calibration ROI (that lives in params.roi_bounds and feeds ellipse/Q-pixel).
    # area_roi is the legacy single-ROI kept for back-compat JSON loading only;
    # all new code should write to area_rois and read from area_rois.
    area_roi: list = field(default_factory=list)   # LEGACY — single analysis ROI [x0,x1,y0,y1]; back-compat only
    area_rois: dict = field(default_factory=dict)  # {roi_id: [x0,x1,y0,y1]} multi analysis ROIs on THIS scan
    drift: tuple | None = None         # (dx, dy) vs the template (from the drift CSV/plugin)
    cal_checkpoints: dict = field(default_factory=dict)  # {stage: calibration snapshot} —
                                       # pre-<step> baselines so re-testing/re-applying a
                                       # calibration starts from the clean state (no compounding)
    indexing_result: Any = None        # bvm_indexing.IndexingResult from IndexerDialog / index_bvm
    orientation_peaks_result: Any = None  # orientation_peaks.OrientationPeaksResult (py4DSTEM GUI)

    def ensure_state(self) -> Any:
        if self.state is None:
            self.state = _new_state()
            if self.raw_path:
                self.state.raw_mib_path = self.raw_path
            if self.braggpeaks_path:
                self.state.braggpeaks_path = self.braggpeaks_path
        return self.state


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_braggpeaks(scan: Scan, path: str | None = None, *, log: Log = None) -> None:
    """Load a pre-computed braggpeaks.h5 into the scan state (cell 21)."""
    st = scan.ensure_state()
    p = path or scan.braggpeaks_path
    if not p or not Path(p).is_file():
        raise FileNotFoundError(f"braggpeaks file not found for '{scan.name}': {p}")
    bp = _pipeline().load_braggpeaks_file(p, log=log)
    st.braggpeaks = bp
    st.braggpeaks_path = p
    scan.braggpeaks_path = p
    scan.cal_checkpoints = {}          # fresh braggpeaks → drop stale calibration baselines
    _log(log, f"[{scan.name}] braggpeaks loaded")


# ── h5 helpers — LIGHT (h5py only, no py4DSTEM) ────────────────────────────────
# Used for the ADF *preview* and the Files-tree h5-root explorer: we read the
# small virtual-image arrays / structure straight from disk, never the heavy 4D
# payload, so a scan can show its ADF without the (slow) datacube load.
H5_SUFFIXES = (".h5", ".hdf5", ".emd")
_VIMG_KEYS = ("annular_dark_field", "bright_field", "dp_mean", "dp_max")


def scan_h5_path(scan: "Scan") -> str:
    """First h5-like file associated with a scan (for ADF preview / root explore).

    Prefers an explicit ``h5_path``, then the braggpeaks .h5 (which carries the
    virtual-image tree for .mib scans), then the raw file when it is itself an h5
    (a standalone .h5 that holds ADF/BF previews + the 4D data)."""
    for src in (scan.h5_path, scan.braggpeaks_path, scan.raw_path):
        if src and Path(src).suffix.lower() in H5_SUFFIXES and Path(src).is_file():
            return str(src)
    return ""


def scan_size_info(scan: "Scan") -> str:
    """Cheap scan-size string for the Files panel: real-space scan grid (R) and, when
    known, the diffraction size (Q). Uses in-memory data first, then peeks the h5
    structure (no heavy load). Returns e.g. ``'R 512×512 · Q 256×256'`` or ''."""
    R = Q = None
    a = cached_adf(scan)                       # ADF already in memory → R = its shape
    if a is not None and getattr(a, "ndim", 0) == 2:
        R = (int(a.shape[0]), int(a.shape[1]))
    st = getattr(scan, "state", None)
    bp = getattr(st, "braggpeaks", None) if st is not None else None
    dc = getattr(st, "datacube", None) if st is not None else None
    for obj in (dc, bp):                       # Q (and R) from loaded objects
        qs = getattr(obj, "Qshape", None)
        if qs is not None and len(qs) >= 2:
            Q = (int(qs[0]), int(qs[1]))
        rs = getattr(obj, "Rshape", None)
        if R is None and rs is not None and len(rs) >= 2:
            R = (int(rs[0]), int(rs[1]))
    if R is None or Q is None:                 # peek the h5 structure (cheap, no payload)
        h5 = scan_h5_path(scan)
        if h5:
            try:
                tree = explore_h5(h5)

                def _walk(n):
                    for ch in (n.get("children") or []):
                        yield ch
                        yield from _walk(ch)
                imgs, dps = [], []
                for ch in _walk(tree or {}):
                    nm = (ch.get("name") or "").lower()
                    shp = ch.get("shape")
                    if not (shp and len(shp) == 2):
                        continue
                    if any(k in nm for k in ("dark_field", "bright_field", "adf", "bf")):
                        imgs.append(tuple(int(s) for s in shp))
                    elif "dp_" in nm or "dp" == nm or "mean" in nm or "max" in nm:
                        dps.append(tuple(int(s) for s in shp))
                if R is None and imgs:
                    R = imgs[0]
                if Q is None and dps:
                    Q = dps[0]
            except Exception:
                pass
    parts = []
    if R:
        parts.append(f"R {R[0]}×{R[1]}")
    if Q:
        parts.append(f"Q {Q[0]}×{Q[1]}")
    return " · ".join(parts)


def explore_h5(path: str, *, max_depth: int = 12) -> dict | None:
    """Nested description of an h5 file's root for the Files-tree explorer.

    Returns ``{name, kind, h5path, shape, dtype, children}`` (kind: group|dataset)
    or None if the file isn't a readable h5. Light: reads structure only (h5py),
    never the heavy 4D payload, so it is cheap to expand on demand."""
    import h5py
    p = Path(path)
    if p.suffix.lower() not in H5_SUFFIXES or not p.is_file():
        return None

    def node(name, obj, h5path, depth):
        if isinstance(obj, h5py.Dataset):
            return {"name": name, "kind": "dataset", "h5path": h5path,
                    "shape": tuple(obj.shape), "dtype": str(obj.dtype), "children": []}
        children = []
        if depth < max_depth:
            try:
                keys = list(obj.keys())
            except Exception:
                keys = []
            for k in keys:
                try:
                    children.append(
                        node(k, obj[k], f"{h5path}/{k}" if h5path else k, depth + 1))
                except Exception:
                    pass
        return {"name": name, "kind": "group", "h5path": h5path,
                "shape": None, "dtype": None, "children": children}

    try:
        with h5py.File(p, "r") as f:
            return node(p.name, f, "", 0)
    except Exception:
        return None


def read_h5_node(path: str, h5path: str) -> "np.ndarray | None":
    """Read one h5 node as an ndarray: a Dataset directly, or a py4DSTEM-style
    group's ``data`` child. Returns None if the node isn't array-like."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            obj = f[h5path] if h5path else f
            if isinstance(obj, h5py.Dataset):
                return np.asarray(obj)
            if hasattr(obj, "keys") and "data" in obj.keys():
                return np.asarray(obj["data"])
    except Exception:
        return None
    return None


def _read_virtual_images_lenient(path: str, *, log: Log = None) -> dict:
    """Read whatever virtual images (ADF/BF/dp_*) an h5 carries WITHOUT requiring
    the full set. Walks for groups/datasets named like the known keys with 2D data.

    py4DSTEM's strict reader (``_load_visualcube_from_h5``) needs ALL FOUR images;
    many companion files carry only ADF + BF, which is exactly why the strict path
    returned nothing and the ADF preview stayed blank. This light fallback unblocks
    the preview for those files. Returns ``{key: 2D float32 ndarray}``."""
    import h5py
    out: dict = {}
    p = Path(path)
    if p.suffix.lower() not in H5_SUFFIXES or not p.is_file():
        return out
    try:
        with h5py.File(p, "r") as f:
            def visit(name, obj):
                base = name.split("/")[-1]
                if base not in _VIMG_KEYS or base in out:
                    return
                arr = None
                if isinstance(obj, h5py.Group) and "data" in obj:
                    arr = np.asarray(obj["data"])
                elif isinstance(obj, h5py.Dataset):
                    arr = np.asarray(obj)
                if arr is not None and arr.ndim == 2:
                    out[base] = arr.astype(np.float32)
            f.visititems(visit)
    except Exception as exc:
        _log(log, f"lenient h5 virtual-image read failed for {p.name}: {exc}")
    return out


def set_cached_adf(scan: "Scan", arr) -> "np.ndarray | None":
    """Store a 2D array as the scan's ADF preview (used by the h5-root explorer
    when the user double-clicks an image node). Returns the stored array or None."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        return None
    st = scan.ensure_state()
    vi = getattr(st, "virtual_images", None) or {}
    vi["annular_dark_field"] = a
    st.virtual_images = vi
    scan.adf_cache = a                 # persist across state resets
    return a


def bragg_vector_map(scan: Scan, *, sampling: int = 1, log: Log = None) -> "np.ndarray | None":
    """The Bragg vector map (BVM) = ``braggpeaks.histogram(mode='raw', sampling=…)``,
    a 2D diffraction-space map of all detected peaks — the right image to pick the
    Origin center on (NOT the real-space ADF). Loads braggpeaks (light, Path A) if
    not already in the state. Returns the 2D ndarray, or None."""
    st = scan.ensure_state()
    bp = getattr(st, "braggpeaks", None)
    if bp is None:
        load_braggpeaks(scan, log=log)
        bp = getattr(st, "braggpeaks", None)
    if bp is None:
        return None
    bvm = bp.histogram(mode="raw", sampling=int(max(1, sampling)))
    arr = np.asarray(getattr(bvm, "data", bvm), dtype=np.float32)
    return arr if arr.ndim == 2 else None


def load_adf(scan: Scan, *, log: Log = None) -> "np.ndarray | None":
    """Load the precomputed virtual images (ADF + BF + dp_mean/max) into the scan.

    The ADF *preview* comes from the .h5 — NOT the heavy .mib datacube — so a scan
    can show its ADF without the heavy load. Source order: ``h5_path →
    braggpeaks_path → raw_path`` (whichever is an h5). For each we try py4DSTEM's
    strict reader first (proper Array objects, all four images); if that fails
    (e.g. the file carries only ADF + BF) we fall back to a light h5py read so the
    preview still works. Returns the ADF ndarray, or None.
    """
    st = scan.ensure_state()
    pl = _pipeline()
    for src in (scan.h5_path, scan.braggpeaks_path, scan.raw_path):
        if (not src or Path(src).suffix.lower() not in H5_SUFFIXES
                or not Path(src).is_file()):
            continue
        sp = Path(src)
        # 1) strict: full virtual-image set as py4DSTEM Array objects (all four)
        try:
            vc = pl._load_visualcube_from_h5(sp, log=log)
        except Exception:
            vc = None
        if vc is not None:
            images: dict = {}
            for key in pl.VIRTUAL_IMAGE_KEYS:
                try:
                    images[key] = pl._read_tree(vc, key)
                except Exception:
                    pass
            adf_node = images.get("annular_dark_field")
            if adf_node is not None:
                st.visualcube = vc
                st.virtual_images = images
                _log(log, f"[{scan.name}] virtual images {list(images)} from {sp.name}")
                arr = np.asarray(getattr(adf_node, "data", adf_node), dtype=np.float32)
                if arr.ndim == 2:
                    scan.adf_cache = arr            # persist for the pickers
                    return arr
                return None
        # 2) lenient: read whatever images are present directly (ADF + BF only is OK)
        images = _read_virtual_images_lenient(sp, log=log)
        if images.get("annular_dark_field") is not None:
            vi = getattr(st, "virtual_images", None) or {}
            vi.update(images)
            st.virtual_images = vi
            _log(log, f"[{scan.name}] virtual images {list(images)} from {sp.name} (lenient)")
            arr = np.asarray(images["annular_dark_field"], dtype=np.float32)
            if arr.ndim == 2:
                scan.adf_cache = arr                # persist for the pickers
                return arr
            return None
    # 3) last resort: the sidecar ADF helper (raw stem → <stem>.h5)
    for src in (scan.h5_path, scan.braggpeaks_path, scan.raw_path):
        if not src:
            continue
        try:
            arr = pl.try_load_adf_from_sidecar_h5(scan.raw_path or src, h5_path=src, log=log)
        except Exception:
            arr = None
        if arr is not None:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                vi = getattr(st, "virtual_images", None) or {}
                vi["annular_dark_field"] = arr
                st.virtual_images = vi
                scan.adf_cache = arr                # persist for the pickers
                return arr
    return None


def cached_adf(scan: Scan) -> "np.ndarray | None":
    """Return the ADF already in memory for this scan, WITHOUT loading from disk.

    Reads the Scan-level ``adf_cache`` FIRST — it survives WorkflowState resets, so
    the picking steps (6 points / ROI / center) keep their image even after a heavy
    datacube (.mib) load wipes ``state.virtual_images``. Falls back to the state's
    virtual images. ``load_adf`` populates both.
    """
    arr = getattr(scan, "adf_cache", None)
    if arr is not None:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 2:
            return a
    st = scan.state
    if st is None:
        return None
    vi = getattr(st, "virtual_images", None) or {}
    node = vi.get("annular_dark_field")
    if node is None:
        return None
    arr = getattr(node, "data", node)
    if getattr(arr, "ndim", 0) != 2:
        return None
    a = np.asarray(arr, dtype=np.float32)
    scan.adf_cache = a                 # promote to the persistent cache
    return a


def find_sidecar_braggpeaks(raw_path: str) -> str | None:
    """Locate a sibling braggpeaks file for a raw scan in the SAME folder.

    The companion detection tool writes ``<stem>braggpeaks.h5`` next to the data
    (e.g. ``Scan04_512.mib`` → ``Scan04_512braggpeaks.h5``). We match that exact
    convention first, then a stem-prefixed variant — never a folder-wide glob, so
    multi-scan folders don't cross-link.
    """
    p = Path(raw_path)
    folder = p.parent
    if not folder.is_dir():
        return None
    exact = folder / f"{p.stem}braggpeaks.h5"
    if exact.is_file():
        return str(exact)
    for g in sorted(folder.glob(f"{p.stem}*braggpeaks*.h5")):
        return str(g)
    return None


def find_sidecar_h5(raw_path: str) -> str:
    """Locate the virtual-images .h5 sibling of a raw scan (the light preview file).

    Convention (matches ``pipeline._default_sidecar_h5_path`` and the user's data):
    a ``Scan01.mib`` raw is paired with ``Scan01.h5`` — same stem, ``.h5`` extension
    — which holds ADF / BF / DP mean / DP max. For a raw that is itself an h5/emd we
    prefer a ``<stem>_precomputed.h5`` sibling, else the file itself. The big ``.mib``
    is NOT touched here; this is purely the cheap preview source. Returns "" if none.
    """
    if not raw_path:
        return ""
    p = Path(raw_path)
    if not p.name:                       # "" / "." → no stem to pair (e.g. workspace scans)
        return ""
    suf = p.suffix.lower()
    if suf in (".h5", ".hdf5", ".emd"):
        sib = p.with_name(f"{p.stem}_precomputed.h5")
        if sib.is_file():
            return str(sib)
        return str(p) if p.is_file() else ""
    cand = p.with_suffix(".h5")
    return str(cand) if cand.is_file() else ""


def calibration_h5_path(raw_path: str, *, explicit: str = "") -> str:
    """Path to the virtual-images / calibration .h5 for a raw scan (``<stem>.h5``).

    Never returns a braggpeaks file — that belongs in ``braggpeaks_path`` only.
    """
    if explicit:
        ep = Path(explicit)
        if ep.is_file() and "braggpeaks" not in ep.name.lower():
            return str(ep.resolve())
    side = find_sidecar_h5(raw_path)
    if side and "braggpeaks" not in Path(side).name.lower():
        return side
    if raw_path:
        cand = Path(raw_path).with_suffix(".h5")
        if cand.is_file() and "braggpeaks" not in cand.name.lower():
            return str(cand.resolve())
    return ""


WORKSPACE_PARAMS_JSON_NAMES = (
    "Parametros_cal.json",
    "fast4d_session.json",
    "params.json",
)


def find_workspace_params_json(root: str | Path) -> Path | None:
    """Locate a params / session JSON beside a saved-workspace batch folder."""
    root = Path(root).resolve()
    for name in WORKSPACE_PARAMS_JSON_NAMES:
        p = root / name
        if p.is_file():
            return p
    return None


def resolve_stored_path(stored: str, *, json_dir: Path, scan_name: str = "") -> str:
    """Resolve a path from JSON — use as-is if it exists, else search near the batch."""
    if not stored:
        return ""
    p = Path(stored)
    if p.is_file():
        return str(p.resolve())
    bases = [json_dir, json_dir.parent]
    if scan_name:
        bases.extend([json_dir / scan_name, json_dir.parent / scan_name])
    seen: set = set()
    for base in bases:
        for cand in (base / p.name, base / p.stem if p.suffix else base):
            cs = str(cand)
            if cs in seen:
                continue
            seen.add(cs)
            if Path(cand).is_file():
                return str(Path(cand).resolve())
    return str(stored)


def _lookup_scan_meta(by_name: dict, scan: "Scan"):
    """Match a hydrated workspace scan to an entry from params/session JSON."""
    if scan.name in by_name:
        return by_name[scan.name]
    per = {k: v for k, v in by_name.items()}
    key = _match_scan_key(per, scan)
    if key is not None:
        return by_name[key]
    return None


def _scans_from_params_json(path: Path) -> list["Scan"]:
    """Load scan list + params from Parametros_cal or session JSON."""
    try:
        return load_session_json(str(path))
    except Exception:
        pass
    try:
        return scans_from_template(str(path))
    except Exception:
        return []


def apply_workspace_params_json(scans: list["Scan"], json_path: str | Path, *,
                                log: Log = None) -> int:
    """Apply Parametros_cal / session JSON to hydrated workspace scans.

    Sets ``raw_path``, ``braggpeaks_path``, calibration ``h5_path`` (virtual
    images — same stem as the raw file, NOT braggpeaks), ``params``, lines and ROI.
    """
    jp = Path(json_path)
    if not jp.is_file():
        return 0
    meta = _scans_from_params_json(jp)
    if not meta:
        _log(log, f"Workspace params JSON not readable: {jp.name}")
        return 0
    by_name = {s.name: s for s in meta}
    json_dir = jp.parent
    applied = 0
    for sc in scans:
        src = _lookup_scan_meta(by_name, sc)
        if src is None:
            continue
        sc.raw_path = resolve_stored_path(src.raw_path or "", json_dir=json_dir,
                                           scan_name=sc.name)
        sc.braggpeaks_path = resolve_stored_path(
            src.braggpeaks_path or "", json_dir=json_dir, scan_name=sc.name)
        explicit_h5 = getattr(src, "h5_path", "") or ""
        if explicit_h5:
            explicit_h5 = resolve_stored_path(explicit_h5, json_dir=json_dir,
                                              scan_name=sc.name)
        sc.h5_path = calibration_h5_path(sc.raw_path, explicit=explicit_h5)
        sc.params = src.params
        sc.params_source = str(jp)
        if getattr(src, "vacuum_path", ""):
            sc.vacuum_path = resolve_stored_path(src.vacuum_path, json_dir=json_dir,
                                                 scan_name=sc.name)
        if getattr(src, "lines", None):
            sc.lines = dict(src.lines)
        if getattr(src, "area_roi", None):
            sc.area_roi = list(src.area_roi)
        if getattr(src, "area_rois", None):
            sc.area_rois = {str(k): list(v) for k, v in src.area_rois.items()}
        applied += 1
        _log(log, f"[{sc.name}] paths from {jp.name}: "
                  f"h5={Path(sc.h5_path).name if sc.h5_path else '—'} "
                  f"({'OK' if sc.h5_path and Path(sc.h5_path).is_file() else 'missing'})")
    try:
        load_lines_json(str(jp), scans, log=log)
    except Exception as exc:
        _log(log, f"Lines from {jp.name} skipped: {exc}")
    try:
        load_roi_json(str(jp), scans, log=log)
    except Exception as exc:
        _log(log, f"ROI from {jp.name} skipped: {exc}")
    _log(log, f"Workspace params: {applied}/{len(scans)} scan(s) from {jp.name}")
    return applied


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE CAPTURE — grab every figure produced by py4DSTEM (pyplot) or the GUI
# into the scan's figure registry, for the multi-scan Report.
# ─────────────────────────────────────────────────────────────────────────────

import contextlib


@contextlib.contextmanager
def capture_pyplot_figures():
    """Context manager that collects every NEW pyplot figure created in the block.

    py4DSTEM plots into the global pyplot state; this snapshots the figure
    numbers before/after and yields a list that, on exit, holds the new figures.

        with capture_pyplot_figures() as figs:
            crystal.calibrate_pixel_size(...)   # py4DSTEM draws internally
        # figs now holds the matplotlib Figure(s) py4DSTEM created
    """
    import matplotlib.pyplot as plt
    before = set(plt.get_fignums())
    grabbed: list = []
    try:
        yield grabbed
    finally:
        new = sorted(set(plt.get_fignums()) - before)
        grabbed.extend(plt.figure(n) for n in new)
        for fig in grabbed:
            try:
                plt.close(fig)  # detach from pyplot; Figure object remains usable by Qt/savefig
            except Exception:
                pass
        if _figure_policy.mode == "off" and _figure_policy.close_orphans:
            grabbed.clear()


def register_figure(scan: "Scan", key: str, fig, *, force: bool = False) -> bool:
    """Register a matplotlib Figure into ``scan.figures[key]`` (Report).

    Respects the active :class:`FigurePolicy` unless ``force=True`` (explicit Commit /
    Compute with figure_mode=report). Returns True when the figure was kept."""
    if fig is None or not hasattr(fig, "savefig"):
        return False
    pol = _figure_policy
    if not force:
        if pol.mode == "off":
            _close_figure(fig)
            return False
        if pol.mode == "preview":
            return False                    # ephemeral — caller displays then closes
        if not pol.store.get(key, True):
            _close_figure(fig)
            return False
    old = scan.figures.get(key)
    if old is not None and old is not fig:
        _close_figure(old)
    scan.figures[key] = fig
    _close_figure(fig)  # keep the Figure object, but remove any pyplot manager/window
    _enforce_figure_ram_limit(scan)
    return True


# Per-step persistence defaults for figure_mode=report (Compute + explicit Commits).
DEFAULT_STORE_FIGURE: dict[str, bool] = {
    "probe": True,
    "select6": False,
    "detection": False,
    "roi": False,
    "origin": True,
    "ellipse": True,
    "q_pixel": True,
    "basis": True,
    "indexing": True,
    "strain_without_roi": True,
    "strain_with_roi": True,
    "stress_without_roi": True,
    "stress_with_roi": True,
    "lines": True,
    "line_profiles": True,
}

# Human labels for the Figure store dialog (same keys as DEFAULT_STORE_FIGURE).
STORE_FIGURE_LABELS: dict[str, str] = {
    "probe": "Probe",
    "select6": "6-point detection",
    "detection": "Detection preview",
    "roi": "ROI snapshot",
    "origin": "Origin",
    "ellipse": "Ellipse",
    "q_pixel": "Q-pixel",
    "basis": "Basis",
    "indexing": "BVM indexing",
    "strain_without_roi": "Strain — Theoretical reference (without ROI)",
    "strain_with_roi": "Strain — Experimental reference (with ROI)",
    "stress_without_roi": "Stress — Theoretical reference (without ROI)",
    "stress_with_roi": "Stress — Experimental reference (with ROI)",
    "lines": "Line map",
    "line_profiles": "Line profiles",
}

# User-facing names for strain/stress reference maps (keys stay without_roi / with_roi).
ROI_REF_LABELS: dict[str, str] = {
    "without_roi": "Theoretical reference (without ROI)",
    "with_roi": "Experimental reference (with ROI)",
}


def roi_ref_label(key: str) -> str:
    """Pretty title for a without_roi / with_roi map label."""
    return ROI_REF_LABELS.get(str(key), str(key))


@dataclass
class FigurePolicy:
    """Session / Compute policy for matplotlib figure retention."""
    mode: str = "report"              # "off" | "preview" | "report"
    store: dict = field(default_factory=lambda: dict(DEFAULT_STORE_FIGURE))
    max_in_ram: int = 12
    close_orphans: bool = True
    spill_to_disk: bool = True        # evicted figures → PNG before closing
    spill_dpi: int = 72             # DPI for temp / .figure_spill sidecars (GUI viewing)
    save_dpi: int = 300             # DPI for figures/ PNG export on Compute / Save (publication-grade)


_figure_policy = FigurePolicy()


def get_figure_policy() -> FigurePolicy:
    return _figure_policy


def set_figure_policy(*, mode: str | None = None, store: dict | None = None,
                      max_in_ram: int | None = None,
                      close_orphans: bool | None = None,
                      spill_to_disk: bool | None = None,
                      spill_dpi: int | None = None,
                      save_dpi: int | None = None) -> None:
    """Update the global figure policy (GUI session or before Compute)."""
    global _figure_policy
    pol = _figure_policy
    if mode is not None:
        pol.mode = str(mode)
    if store is not None:
        pol.store = dict(store)
    if max_in_ram is not None:
        pol.max_in_ram = max(0, int(max_in_ram))
    if close_orphans is not None:
        pol.close_orphans = bool(close_orphans)
    if spill_to_disk is not None:
        pol.spill_to_disk = bool(spill_to_disk)
    if spill_dpi is not None:
        pol.spill_dpi = max(48, min(240, int(spill_dpi)))
    if save_dpi is not None:
        pol.save_dpi = max(72, min(400, int(save_dpi)))


def pyplot_figure_count() -> int:
    try:
        import matplotlib.pyplot as plt
        return len(plt.get_fignums())
    except Exception:
        return 0


def _close_figure(fig) -> None:
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def close_figure(fig) -> None:
    """Public wrapper — close one matplotlib Figure."""
    _close_figure(fig)


def _registered_figure_set(scans: list | None = None) -> set:
    keep: set = set()
    for sc in scans or []:
        for fig in (getattr(sc, "figures", None) or {}).values():
            if fig is not None and hasattr(fig, "number"):
                keep.add(fig)
    return keep


def _figure_keys(scan: "Scan") -> set[str]:
    spill = getattr(scan, "figure_spill", None) or {}
    return set(scan.figures) | set(spill)


def list_figure_keys(scan: "Scan") -> list[str]:
    """Keys available for the Report browser (in-RAM or spilled) — no Figure load."""
    keys = _figure_keys(scan)
    ordered = [k for k in FIGURE_ORDER if k in keys]
    rest = sorted(k for k in keys if k not in FIGURE_ORDER)
    return ordered + rest


def _spill_dir(scan: "Scan") -> "Path":
    import tempfile
    from pathlib import Path as _P
    if scan.results_dir:
        d = _P(scan.results_dir) / ".figure_spill"
    else:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_"
                       for c in scan.name) or "scan"
        d = _P(tempfile.gettempdir()) / "fast4d" / "spill" / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def spill_figure_to_disk(scan: "Scan", key: str, fig, *, dpi: int | None = None,
                         log: Log = None) -> str | None:
    """Write *fig* to a PNG sidecar and record ``scan.figure_spill[key]``."""
    if fig is None or not hasattr(fig, "savefig"):
        return None
    from pathlib import Path as _P
    dpi = int(_figure_policy.spill_dpi if dpi is None else dpi)
    path = _spill_dir(scan) / f"{key}.png"
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        scan.figure_spill[key] = str(path)
        _log(log, f"[{scan.name}] figure '{key}' spilled → {path.name} (dpi={dpi})")
        return str(path)
    except Exception as exc:
        _log(log, f"[{scan.name}] figure spill failed ({key}): {exc}")
        return None


def _figure_from_png(path: str):
    """Build a matplotlib Figure from a spilled PNG (for Report / enlarge)."""
    from pathlib import Path as _P
    from matplotlib.figure import Figure
    import matplotlib.image as mpimg
    p = _P(path)
    if not p.is_file():
        return None
    try:
        arr = mpimg.imread(str(p))
    except Exception:
        return None
    fig = Figure(figsize=(6.4, 5.0), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.imshow(arr)
    ax.axis("off")
    return fig


def resolve_figure(scan: "Scan", key: str):
    """In-RAM Figure, or reload from ``figure_spill`` PNG if evicted."""
    fig = (scan.figures or {}).get(key)
    if fig is not None:
        return fig
    path = (getattr(scan, "figure_spill", None) or {}).get(key)
    if path:
        return _figure_from_png(path)
    return None


def resolve_figure_path(scan: "Scan", key: str) -> str:
    """PNG path for a spilled figure, or '' if held in RAM only."""
    if (scan.figures or {}).get(key) is not None:
        return ""
    return str((getattr(scan, "figure_spill", None) or {}).get(key) or "")


def clear_figure_spill(scan: "Scan", *, keys: list | None = None,
                       delete_files: bool = True) -> list[str]:
    """Remove spilled PNG references (and optionally delete the files)."""
    from pathlib import Path as _P
    spill = getattr(scan, "figure_spill", None) or {}
    removed: list[str] = []
    for k in list(spill.keys()):
        if keys is not None and k not in keys:
            continue
        path = spill.pop(k, "")
        if delete_files and path:
            try:
                _P(path).unlink(missing_ok=True)
            except Exception:
                pass
        removed.append(k)
    return removed


def close_orphan_pyplot_figures(keep: set | list | None = None) -> int:
    """Close pyplot figures not in *keep*. Returns how many were closed."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return 0
    keep_set = set(keep or [])
    keep_nums = set()
    for fig in keep_set:
        try:
            keep_nums.add(int(fig.number))
        except Exception:
            pass
    closed = 0
    for n in list(plt.get_fignums()):
        if n not in keep_nums:
            plt.close(n)
            closed += 1
    return closed


def _enforce_figure_ram_limit(scan: "Scan") -> None:
    lim = int(_figure_policy.max_in_ram)
    if lim <= 0:
        return
    while len(scan.figures) > lim:
        victim = None
        for k in scan.figures:
            if not _figure_policy.store.get(k, True):
                victim = k
                break
        if victim is None:
            victim = next(iter(scan.figures))
        fig = scan.figures.pop(victim, None)
        if fig is not None:
            if _figure_policy.spill_to_disk:
                spill_figure_to_disk(scan, victim, fig)
            _close_figure(fig)


@dataclass
class ResidentDataPolicy:
    """How many scans' heavy compute buffers (datacube / BVM / probe) may stay
    resident in RAM at once — the same LRU-with-eviction idea FigurePolicy
    already proves out for figures (engine.py:975-1178), applied to the bigger
    objects instead. Eviction here calls the cheap `release_scans` (nulls
    references only, no gc.collect/OS trim — see its own docstring for why),
    not the heavier `free_memory`."""
    max_scans_in_ram: int = 2


_data_policy = ResidentDataPolicy()


def get_data_policy() -> ResidentDataPolicy:
    return _data_policy


def set_data_policy(*, max_scans_in_ram: int | None = None) -> None:
    """Update the global resident-data policy (GUI settings)."""
    global _data_policy
    if max_scans_in_ram is not None:
        _data_policy.max_scans_in_ram = max(1, int(max_scans_in_ram))


def enforce_resident_data_limit(scans: list, active_index: int, recent_indices: list[int],
                                *, log: Log = None) -> list[int]:
    """Update the LRU window to include ``active_index`` first, release every
    scan that falls outside the resulting window (per ``get_data_policy()``),
    and return the new window for the caller to store.

    Pure w.r.t. its inputs aside from the ``release_scans`` side effect — safe
    to call on every scan-switch."""
    limit = get_data_policy().max_scans_in_ram
    new_recent = ([active_index] + [i for i in recent_indices if i != active_index])[:limit]
    to_release = [sc for i, sc in enumerate(scans) if i not in new_recent]
    if to_release:
        release_scans(to_release, log=log)
    return new_recent


@dataclass
class AnalysisScopePolicy:
    """Whether the Report's cross-file views (lines/ROIs "across files", and the
    cross-scan distribution/box/PCA/stress/stats views) may combine every
    currently loaded scan.

    Off by default: two unrelated files loaded in the same session must never
    get silently averaged/compared just because they happen to share a line or
    ROI id like "L1". Turn this on only for a genuine reproducibility experiment
    (repeated measurements of the same sample/line across files)."""
    shared_stats: bool = False


_analysis_scope = AnalysisScopePolicy()


def get_analysis_scope() -> AnalysisScopePolicy:
    return _analysis_scope


def set_analysis_scope(*, shared_stats: bool | None = None) -> None:
    """Update the global analysis-scope policy (Analysis panel checkbox)."""
    global _analysis_scope
    if shared_stats is not None:
        _analysis_scope.shared_stats = bool(shared_stats)


def clear_preview_figures(scan: "Scan") -> list[str]:
    """Drop registered figures marked as non-report (``DEFAULT_STORE_FIGURE`` False)."""
    removed: list[str] = []
    for k in list(scan.figures.keys()):
        if not DEFAULT_STORE_FIGURE.get(k, True):
            fig = scan.figures.pop(k, None)
            if fig is not None:
                _close_figure(fig)
            removed.append(k)
    if removed:
        removed.extend(clear_figure_spill(scan, keys=removed, delete_files=True))
    return removed


def clear_all_figures(scan: "Scan", *, keep_report: bool = True) -> list[str]:
    """Clear registered figures. When *keep_report*, retain store=True keys."""
    removed: list[str] = []
    for k in list(scan.figures.keys()):
        if keep_report and _figure_policy.store.get(k, DEFAULT_STORE_FIGURE.get(k, True)):
            continue
        fig = scan.figures.pop(k, None)
        if fig is not None:
            _close_figure(fig)
        clear_figure_spill(scan, keys=[k], delete_files=True)
        removed.append(k)
    return removed


def figure_memory_status(scans: list | None = None) -> str:
    n_mpl = pyplot_figure_count()
    n_reg = sum(len(getattr(s, "figures", None) or {}) for s in (scans or []))
    n_spill = sum(len(getattr(s, "figure_spill", None) or {}) for s in (scans or []))
    pol = _figure_policy
    spill_note = f" · {n_spill} on disk" if n_spill else ""
    return (f"{n_mpl} matplotlib window(s) · {n_reg} in RAM{spill_note} · "
            f"mode={pol.mode} · max={pol.max_in_ram}"
            + (" · spill=ON" if pol.spill_to_disk else "")
            + f" · view={pol.spill_dpi}dpi · save={pol.save_dpi}dpi")


def tidy_figure_memory(scans: list | None = None, *, log: Log = None) -> dict:
    """Close orphan pyplot figures + optional preview keys. Safe after tool previews."""
    scans = list(scans or [])
    prev = 0
    for sc in scans:
        prev += len(clear_preview_figures(sc))
    orphans = 0
    if _figure_policy.close_orphans:
        orphans = close_orphan_pyplot_figures(_registered_figure_set(scans))
    msg = figure_memory_status(scans)
    if prev or orphans:
        _log(log, f"Figure tidy: dropped {prev} preview slot(s), closed {orphans} orphan(s). {msg}")
    return {"preview_removed": prev, "orphans_closed": orphans, "status": msg}


# Canonical figure order for the Report (detection → calibration → maps).
FIGURE_ORDER = (
    "probe", "select6", "detection",
    "roi", "origin", "ellipse", "q_pixel", "basis", "indexing",
    "strain_without_roi", "strain_with_roi",
    "stress_without_roi", "stress_with_roi",
)


def collect_figures(scan: "Scan") -> dict:
    """Ordered {key: Figure} for the Report — in-RAM or reloaded from spill PNGs."""
    keys = _figure_keys(scan)
    out: dict = {}
    for k in FIGURE_ORDER:
        if k in keys:
            fig = resolve_figure(scan, k)
            if fig is not None:
                out[k] = fig
    for k in sorted(keys):
        if k in out:
            continue
        fig = resolve_figure(scan, k)
        if fig is not None:
            out[k] = fig
    return out


VIMG_CMAPS = ("gray", "gray_r", "inferno", "viridis", "magma", "plasma", "bone", "cividis")


def save_virtual_images(scan: "Scan", out_dir: str | "Path", *,
                        cmap: str = "gray", log: Log = None) -> list:
    """Save the virtual images (ADF / BF / DP mean / DP max) as BOTH raw data (.npy)
    and a PLAIN image (.png) — no axes, no colorbar, native resolution (one pixel per
    array element, e.g. 512×512), just the image. ``cmap`` (default 'gray') applies to
    all of them. Reads from state.virtual_images (loading the light .h5 if needed)."""
    import matplotlib.image as mpimg
    from pathlib import Path as _P
    cmap = cmap if cmap in VIMG_CMAPS else "gray"
    st = scan.ensure_state()
    vi = dict(getattr(st, "virtual_images", None) or {})
    if not vi or vi.get("annular_dark_field") is None:
        try:
            load_adf(scan, log=log)            # re-read the light .h5 (ADF/BF/DP…)
            vi = dict(getattr(st, "virtual_images", None) or {})
        except Exception as exc:
            _log(log, f"[{scan.name}] virtual-image reload skipped: {exc}")
    if vi.get("annular_dark_field") is None and getattr(scan, "adf_cache", None) is not None:
        vi["annular_dark_field"] = scan.adf_cache    # cache survives datacube resets

    out = _P(out_dir); out.mkdir(parents=True, exist_ok=True)
    label = {"annular_dark_field": "ADF", "bright_field": "BF",
             "dp_mean": "DP_mean", "dp_max": "DP_max"}
    written: list = []
    for key, name in label.items():
        node = vi.get(key)
        if node is None:
            continue
        arr = np.asarray(getattr(node, "data", node), dtype=float)
        if arr.ndim != 2:
            continue
        np.save(out / f"{name}.npy", arr)            # raw data
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmin = float(np.percentile(finite, 1)); vmax = float(np.percentile(finite, 99))
        else:
            vmin, vmax = float(arr.min()), float(arr.max() or 1.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        mpimg.imsave(str(out / f"{name}.png"), arr, cmap=cmap, vmin=vmin, vmax=vmax,
                     origin="upper")                 # plain image: no axes/colorbar
        written.append(name)
    _log(log, f"[{scan.name}] virtual images saved → {out.name}/ : "
              f"{written or 'none found'} (cmap={cmap})")
    return written


def save_figures(scan: "Scan", out_dir: str | "Path", *, dpi: int | None = None,
                 log: Log = None) -> list:
    """Write every registered figure to ``out_dir`` as PNG. Returns written paths."""
    from pathlib import Path as _P
    dpi = int(_figure_policy.save_dpi if dpi is None else dpi)
    d = _P(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for key, fig in collect_figures(scan).items():
        try:
            p = d / f"{key}.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            written.append(p)
        except Exception as exc:
            _log(log, f"[{scan.name}] could not save figure {key}: {exc}")
    return written


def clean_duplicate_figure_pngs(scan: "Scan", *, log: Log = None) -> list:
    """Remove legacy duplicate PNGs written by the artifact layer.

    ``fast_artifacts`` can write ``strain_strain_without_roi_0.png`` and
    ``strain_strain_with_roi_0.png`` from ``state.strain_figures``. Fast4D also
    writes canonical Report images ``strain_without_roi.png`` / ``strain_with_roi.png``.
    Keep the canonical files and delete the duplicates so the figures folder is clean.
    """
    fdir = Path(getattr(scan, "figures_dir", "") or "")
    if not fdir.is_dir() and getattr(scan, "results_dir", ""):
        fdir = Path(scan.results_dir) / "figures"
    if not fdir.is_dir():
        return []
    removed: list = []
    pairs = {
        "strain_strain_without_roi_0.png": "strain_without_roi.png",
        "strain_strain_with_roi_0.png": "strain_with_roi.png",
    }
    for dup, canonical in pairs.items():
        dp = fdir / dup
        cp = fdir / canonical
        if dp.is_file() and cp.is_file():
            try:
                dp.unlink()
                removed.append(str(dp))
            except Exception as exc:
                _log(log, f"[{scan.name}] could not remove duplicate {dup}: {exc}")
    if removed:
        _log(log, f"[{scan.name}] removed duplicate figure PNG(s): "
                  + ", ".join(Path(p).name for p in removed))
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# Path A vs Path B — single source of truth for GUI + driver gating
# ─────────────────────────────────────────────────────────────────────────────
#   Path A  — braggpeaks.h5 on disk: light load → calibrate → strain (no vacuum/probe)
#   Path B  — no braggpeaks yet: heavy datacube → probe → detect → braggpeaks.h5
# ─────────────────────────────────────────────────────────────────────────────

def analysis_path(scan: "Scan | None") -> str:
    """``'A'`` when a usable ``braggpeaks.h5`` exists; otherwise ``'B'``."""
    if scan is None:
        return "B"
    bp = (scan.braggpeaks_path or "").strip()
    if bp and Path(bp).is_file():
        return "A"
    return "B"


def needs_detection_workflow(scan: "Scan | None") -> bool:
    """True when the heavy Path-B pipeline (datacube → probe → braggpeaks) applies."""
    return analysis_path(scan) == "B"


def ensure_braggpeaks_for_calibration(scan: "Scan", *, log: Log = None) -> None:
    """Light Path-A entry: load ``braggpeaks.h5`` if not already in RAM."""
    st = scan.ensure_state()
    if getattr(st, "braggpeaks", None) is not None:
        return
    if analysis_path(scan) != "A":
        raise RuntimeError(
            f"[{scan.name}] no braggpeaks.h5 — run Path B detection first, or add a braggpeaks file.")
    load_braggpeaks(scan, log=log)


_CALIB_UI_FLAG = {"origin": "origin", "ellipse": "ellipse", "qpixel": "qpx", "basis": "basis"}


def calibration_ui_flags(scan: "Scan") -> dict[str, str]:
    """Per-step calibration strip flags (``applied`` | ``staged`` | ``pending`` | ``unused``)."""
    try:
        from pipeline import single_scan_cal_ui_flags
        return single_scan_cal_ui_flags(scan.ensure_state())
    except Exception:
        return {}


def calibration_step_status(scan: "Scan", step_key: str) -> str:
    """UI status for one calibration step key (origin / ellipse / qpixel / basis)."""
    fk = _CALIB_UI_FLAG.get(step_key)
    if not fk:
        return "pending"
    status = calibration_ui_flags(scan).get(fk, "pending")
    # braggpeaks may still have calstate.ellipse=False from an earlier skip even though
    # the user re-enabled Ellipse in the parameter table — treat as pending, not unused.
    if step_key == "ellipse" and status == "unused" and scan.params.ellipse_enabled:
        return "pending"
    return status


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION (Path B) — raw data → braggpeaks.h5  (OPTIONAL, only when no .h5 yet)
#
#   load_datacube → compute_probe(vacuum) → set_six_points(on ADF) →
#   detect_preview (tune detect_params at the 6 points) → compute_braggpeaks (CUDA)
#
# Heavy: loads the full 4D datacube. The produced braggpeaks.h5 then feeds the
# normal calibration path (origin→…→strain).
# ─────────────────────────────────────────────────────────────────────────────

DETECTION_ORDER = ("probe", "select6", "detection")


def load_datacube(scan: Scan, *, log: Log = None) -> None:
    """Load the raw 4D datacube (cell 7). HEAVY — the big .mib.

    ``load_data_step`` calls ``reset_data_products()`` which CLEARS
    ``state.virtual_images`` (the ADF/BF preview). The picking steps and figures all
    read the ADF, so we re-attach the light virtual images from the .h5 afterwards
    (``scan.adf_cache`` also keeps the ADF independently). This is the .mib↔.h5
    differentiation: the datacube is heavy and transient; the ADF preview persists.
    """
    st = scan.ensure_state()
    _pipeline().load_data_step(
        st, scan.raw_path,
        precomputed_h5_path=(scan.h5_path or None),
        braggpeaks_path=(scan.braggpeaks_path or None),
        use_existing_braggpeaks=bool(scan.braggpeaks_path),
        log=log)
    # re-cache the light virtual images wiped by the heavy load, so 6-pt/ROI/center
    # picking still has an ADF (reads from the .h5 — never the .mib).
    try:
        if not (getattr(st, "virtual_images", None) or {}).get("annular_dark_field"):
            load_adf(scan, log=log)
    except Exception as exc:
        _log(log, f"[{scan.name}] ADF re-cache after datacube load skipped: {exc}")
    _log(log, f"[{scan.name}] datacube loaded")


# ─────────────────────────────────────────────────────────────────────────────
# VIRTUALIZATION — build the virtual-images .h5 (ADF/BF/DP mean/max) from the raw
# datacube. Port of the single-mode virtualization_window.py / virtualizationAlgorithm.
#   get_dp_mean / get_dp_max / get_probe_size → set detectors → get_virtual_image
#   (annular=ADF, circle=BF) → py4DSTEM.save(..., tree=None, mode='o')
# UI-free; heavy passes run in the GUI's worker. No pyplot.
# ─────────────────────────────────────────────────────────────────────────────

def vc_compute_dp_probe(scan: Scan, *, mode: str = "both", log: Log = None) -> dict:
    """Load the raw datacube (if needed) and compute DP mean and/or max + probe
    center. ``mode`` selects which full-4D pass(es) to run — 'mean' (dp_max
    stays None), 'max' (dp_mean stays None), or 'both' (default, original
    behavior). The probe center/alpha is derived from dp_mean when available,
    else dp_max — one of the two is always present.

    Returns ``{dp_mean, dp_max, alpha, qx0, qy0, qmax}`` (qx0=row, qy0=col). Stores the
    probe centre on the state (so the OriginDialog can start there too)."""
    m = (mode or "both").lower()
    if m not in ("mean", "max", "both"):
        raise ValueError(f"mode must be 'mean', 'max' or 'both' (got {mode!r}).")
    st = scan.ensure_state()
    if getattr(st, "datacube", None) is None:
        load_datacube(scan, log=log)              # heavy .mib load
    cube = getattr(st, "datacube", None)
    if cube is None:
        raise RuntimeError("No raw datacube — this scan needs a raw .mib/.dm4/.h5 path.")
    dp_mean = dp_max = None
    if m in ("mean", "both"):
        _log(log, f"[{scan.name}] DP mean (full 4D pass)…")
        dp_mean = cube.get_dp_mean()
    if m in ("max", "both"):
        _log(log, f"[{scan.name}] DP max…")
        dp_max = cube.get_dp_max()
    probe_src = dp_mean if dp_mean is not None else dp_max
    _log(log, f"[{scan.name}] probe size from DP {'mean' if dp_mean is not None else 'max'}…")
    alpha, qx0, qy0 = cube.get_probe_size(probe_src.data)
    st.probe_alpha = float(alpha); st.probe_qx0 = float(qx0); st.probe_qy0 = float(qy0)
    qshape = getattr(cube, "Qshape", None)
    qmax = float(min(qshape) / 2.0) if qshape else 256.0
    return {"dp_mean": (np.asarray(dp_mean.data, dtype=np.float32) if dp_mean is not None else None),
            "dp_max": (np.asarray(dp_max.data, dtype=np.float32) if dp_max is not None else None),
            "alpha": float(alpha), "qx0": float(qx0), "qy0": float(qy0), "qmax": qmax}


def vc_compute_virtual_images(scan: Scan, *, which: str, center_yx, adf_radii,
                              bf_radius: float, log: Log = None) -> None:
    """Compute the ADF (annular) and/or BF (circle) virtual images on the datacube
    tree (``get_virtual_image``). ``center_yx`` = (row, col); ``adf_radii`` = (inner,
    outer). ``which`` ∈ {'adf','bf','both'}."""
    import gc
    st = scan.ensure_state()
    cube = getattr(st, "datacube", None)
    if cube is None:
        raise RuntimeError("Load the datacube first (vc_compute_dp_probe).")
    w = (which or "both").lower()
    c = (int(round(center_yx[0])), int(round(center_yx[1])))
    if w in ("adf", "both"):
        r0, r1 = float(adf_radii[0]), float(adf_radii[1])
        if not (r1 > r0 > 0):
            raise ValueError("ADF radii must satisfy outer > inner > 0.")
        _log(log, f"[{scan.name}] ADF annular center(y,x)={c} radii={(r0, r1)}…")
        cube.get_virtual_image(mode="annular", geometry=(c, (r0, r1)),
                               name="annular_dark_field")
        gc.collect()
    if w in ("bf", "both"):
        if not (float(bf_radius) > 0):
            raise ValueError("BF radius must be > 0.")
        _log(log, f"[{scan.name}] BF circle center(y,x)={c} radius={float(bf_radius)}…")
        cube.get_virtual_image(mode="circle", geometry=(c, float(bf_radius)),
                               name="bright_field")
        gc.collect()
    _log(log, f"[{scan.name}] virtual images computed ({w}).")


def vc_read_virtual(scan: Scan, key: str):
    """Read a computed virtual image (annular_dark_field / bright_field / dp_mean /
    dp_max) from the datacube tree → 2D float array (or None)."""
    st = scan.ensure_state()
    cube = getattr(st, "datacube", None)
    if cube is None:
        return None
    node = None
    try:
        node = cube.tree(key)
    except Exception:
        try:
            node = cube.tree("datacube_root").tree(key)
        except Exception:
            node = None
    if node is None:
        return None
    return np.asarray(getattr(node, "data", node), dtype=float)


def vc_save_h5(scan: Scan, path: str, *, log: Log = None) -> str:
    """Save DP mean/max + ADF/BF onto an .h5.

    When the target is the same EMD/HDF5 the cube was loaded from (tutorial /
    simulation files), append/overwrite only the virtual-image children under
    the existing DataCube path (``mode='ao'`` + ``emdpath``) so sibling groups
    (polyAu, vacuum_probe, …) are preserved. New sidecar files for .mib scans
    still use overwrite (``mode='o'``, ``tree=None``) like the notebook.
    """
    import py4DSTEM
    st = scan.ensure_state()
    cube = getattr(st, "datacube", None)
    if cube is None:
        raise RuntimeError("No datacube to save.")

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    emd_dp = (
        getattr(st, "emd_datapath", None)
        or getattr(cube, "_fast4d_emd_datapath", None)
        or None
    )
    if isinstance(emd_dp, str):
        emd_dp = emd_dp.strip() or None

    raw = Path(scan.raw_path).expanduser() if scan.raw_path else None
    same_source = (
        out.exists()
        and raw is not None
        and raw.suffix.lower() in (".h5", ".hdf5", ".emd")
        and out.resolve() == raw.resolve()
    )
    # Also append when writing back into an existing multi-cube EMD we know the path for.
    append_in_place = bool(emd_dp) and (same_source or (out.exists() and out.suffix.lower() in (".h5", ".hdf5", ".emd")))

    if append_in_place and emd_dp:
        _log(log, f"[{scan.name}] appending virtual images into {out} @ {emd_dp}")
        py4DSTEM.save(str(out), cube, tree=None, mode="ao", emdpath=str(emd_dp))
    else:
        _log(log, f"[{scan.name}] saving virtual-images h5 → {out}")
        py4DSTEM.save(str(out), cube, tree=None, mode="o")

    scan.h5_path = str(out)
    try:
        st.precomputed_h5_path = out
    except Exception:
        pass
    _log(log, f"[{scan.name}] virtual-images h5 saved.")
    return str(out)


def compute_probe(scan: Scan, *, source: str | None = None, log: Log = None) -> None:
    """Compute the probe/template. Source: vacuum | bf_roi | synthetic | mean_dp."""
    pl = _pipeline()
    st = scan.ensure_state()
    src = (source or scan.params.probe_source or "vacuum").lower()
    if src == "vacuum":
        if not scan.vacuum_path:
            raise RuntimeError(f"[{scan.name}] no vacuum_path set for probe.")
        pl.compute_probe_step(st, scan.vacuum_path, log=log)
    elif src == "bf_roi":
        # The vacuum-ROI probe averages real DPs over the region → it needs the 4D
        # datacube (the heavy .mib). The ROI itself is picked on the light .h5 ADF,
        # so the cube isn't loaded until this point; load it on demand now (this is
        # the file <-> file connection: same Scan carries raw_path + h5_path).
        if getattr(st, "datacube", None) is None:
            if not scan.raw_path:
                raise RuntimeError(
                    f"[{scan.name}] vacuum-ROI probe needs the raw 4D data (.mib), but no "
                    f"raw_path is set. Use a separate vacuum file instead (probe source "
                    f"'vacuum'), or add the .mib for this scan.")
            _log(log, f"[{scan.name}] vacuum-ROI probe needs the 4D datacube — loading the "
                      f"raw ({Path(scan.raw_path).name}) now; this is the heavy load.")
            load_datacube(scan, log=log)
        if scan.params.probe_vacuum_roi_bounds:      # apply the picked ROI (now has datacube)
            pl.set_probe_bf_vacuum_roi_from_bounds(
                st, tuple(int(v) for v in scan.params.probe_vacuum_roi_bounds), log=log)
        pl.compute_probe_from_bf_vacuum_roi_step(st, log=log)
    elif src == "synthetic":
        pl.compute_synthetic_disk_probe_step(st, log=log)
    elif src == "mean_dp":
        pl.compute_probe_from_mean_dp_patch_step(st, log=log)
    else:
        raise ValueError(f"Unknown probe source: {src}")
    scan.params.probe_source = src
    # capture a probe-kernel figure for the Report
    try:
        kernel = pl.probe_kernel_template_ndarray(st.probe)
        if kernel is not None:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(3.4, 3.4))
            ax = fig.add_subplot(111)
            ax.imshow(np.asarray(kernel), cmap="inferno", origin="upper")
            ax.set_title(f"Probe kernel ({src})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            register_figure(scan, "probe", fig)
    except Exception as exc:
        _log(log, f"[{scan.name}] probe figure skipped: {exc}")
    _log(log, f"[{scan.name}] probe ready (source={src})")


def compute_shared_probe(vacuum_path: str, *, log: Log = None):
    """Compute a probe template from a vacuum FILE on a throwaway state.

    The vacuum is shared across all scans, so the probe is computed once here and
    can be applied to every scan via ``apply_probe``. Returns the py4DSTEM probe.
    """
    st = _new_state()
    _pipeline().compute_probe_step(st, vacuum_path, log=log)
    return getattr(st, "probe", None)


def apply_probe(scan: Scan, probe, *, source: str = "vacuum", log: Log = None) -> None:
    """Attach an already-computed (shared) probe to a scan's state."""
    if probe is None:
        return
    st = scan.ensure_state()
    st.probe = probe
    scan.params.probe_source = source
    _log(log, f"[{scan.name}] shared probe applied (source={source})")


def set_probe_vacuum_roi(scan: Scan, bounds, *, log: Log = None) -> None:
    """Mark a real-space rectangle on the sample's own vacuum zone as the probe ROI.

    This is the "pick-up region for vacuum" from the legacy GUI: the probe template
    is built by averaging DPs over this region of the MAIN datacube (needs the
    datacube loaded). Sets ``probe_source='bf_roi'`` so ``compute_probe`` uses it.
    """
    scan.params.probe_vacuum_roi_bounds = [int(v) for v in bounds]
    scan.params.probe_source = "bf_roi"
    st = scan.ensure_state()
    if getattr(st, "datacube", None) is not None:     # apply now if data is loaded
        try:
            _pipeline().set_probe_bf_vacuum_roi_from_bounds(
                st, tuple(int(v) for v in bounds), log=log)
        except Exception as exc:
            _log(log, f"[{scan.name}] probe ROI will apply at compute: {exc}")
    _log(log, f"[{scan.name}] vacuum ROI stored {scan.params.probe_vacuum_roi_bounds} "
              f"(probe_source=bf_roi; Compute probe needs the datacube loaded)")


def _fill_probe_kernel_panels(kernel2d, ax_tile, ax_prof, R=20, L=20, W=1) -> None:
    """Notebook-style kernel tiling + dual line profiles (mirrors py4DSTEM show_kernel)."""
    k = np.asarray(kernel2d, dtype=float)
    if k.ndim != 2:
        ax_tile.text(0.02, 0.98, "Kernel not 2D", va="top", transform=ax_tile.transAxes)
        ax_prof.axis("off")
        return
    Ry, Rx = k.shape
    if Ry < 4 or Rx < 4:
        ax_tile.imshow(k, cmap="gray"); ax_tile.set_title("Probe kernel"); ax_tile.axis("off")
        ax_prof.axis("off")
        return
    Ri = max(2, min(int(R), Ry // 2, Rx // 2))
    Li = max(1, min(int(L), Ry // 2, Rx // 2))
    Wi = max(1, min(int(W), Rx, Ry))
    lp1 = np.concatenate([np.sum(k[-Li:, :Wi], axis=1), np.sum(k[:Li, :Wi], axis=1)])
    lp2 = np.concatenate([np.sum(k[:Wi, -Li:], axis=0), np.sum(k[:Wi, :Li], axis=0)])
    im_kernel = np.vstack([np.hstack([k[-Ri:, -Ri:], k[-Ri:, :Ri]]),
                           np.hstack([k[:Ri, -Ri:], k[:Ri, :Ri]])])
    ax_tile.matshow(im_kernel, cmap="gray")
    x = np.arange(2 * Ri, dtype=float)
    ax_tile.plot(np.ones(2 * Ri) * Ri, x, c="r", linewidth=1.0)
    ax_tile.plot(x, np.ones(2 * Ri) * Ri, c="c", linewidth=1.0)
    ax_tile.set_title("Kernel (notebook-style tiling)")
    ax_tile.set_xticks([]); ax_tile.set_yticks([])
    ax_prof.plot(np.arange(len(lp1)), lp1, c="r", label="profile 1")
    ax_prof.plot(np.arange(len(lp2)), lp2, c="c", label="profile 2")
    ax_prof.set_title("Kernel line profiles"); ax_prof.set_xlabel("index")
    ax_prof.legend(loc="upper right", fontsize=9)


def build_probe_figure(scan: Scan):
    """Full 4-panel probe figure (notebook style): vacuum probe, full kernel,
    kernel tiling, kernel line profiles. Returns a matplotlib Figure or None.

    Ported from the legacy GUI's ``_build_probe_figure`` so the Probe tab shows
    the same 4 panels py4DSTEM opens.
    """
    from matplotlib.figure import Figure
    st = scan.state
    probe = getattr(st, "probe", None) if st is not None else None
    if probe is None:
        return None
    arr_probe = np.asarray(getattr(probe, "probe", None) if hasattr(probe, "probe") else None)
    if arr_probe is None or arr_probe.size == 0:
        arr_probe = np.asarray(getattr(probe, "data", None) if hasattr(probe, "data") else None)
    if arr_probe is None or arr_probe.size == 0:
        return None
    try:
        kernel = np.asarray(getattr(getattr(probe, "kernel", None), "data", None))
    except Exception:
        kernel = np.asarray([])

    fig = Figure(figsize=(9.2, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.9], width_ratios=[1.0, 1.0])
    ax_probe = fig.add_subplot(gs[0, 0])
    ax_probe.imshow(np.asarray(arr_probe, dtype=float), cmap="inferno")
    ax_probe.set_title("Vacuum probe"); ax_probe.axis("off")
    ax_k2 = fig.add_subplot(gs[0, 1])
    inner = gs[1, :].subgridspec(1, 2, wspace=0.25)
    ax_tile = fig.add_subplot(inner[0, 0])
    ax_prof = fig.add_subplot(inner[0, 1])
    if kernel.size:
        ax_k2.imshow(np.asarray(kernel, dtype=float), cmap="inferno")
        ax_k2.set_title("Probe kernel (full)"); ax_k2.axis("off")
        _fill_probe_kernel_panels(kernel, ax_tile, ax_prof)
    else:
        ax_k2.axis("off"); ax_tile.axis("off"); ax_prof.axis("off")
        ax_k2.text(0.02, 0.98, "Kernel not available", va="top", transform=ax_k2.transAxes)
    register_figure(scan, "probe", fig)
    return fig


def probe_images(scan: Scan) -> list:
    """(title, 2D ndarray) pairs for the Probe results tab: vacuum DP + kernel.

    py4DSTEM's probe object carries ``.probe`` (the averaged vacuum diffraction
    pattern) and ``.kernel`` (the sigmoid detection template).
    """
    st = scan.state
    pr = getattr(st, "probe", None) if st is not None else None
    if pr is None:
        return []
    out: list = []
    img = getattr(pr, "probe", None)
    if img is not None:
        out.append(("Probe (vacuum DP)", np.asarray(img, dtype=float)))
    try:
        ker = _pipeline().probe_kernel_template_ndarray(pr)
        if ker is not None:
            out.append(("Kernel", np.asarray(ker, dtype=float)))
    except Exception:
        pass
    return out


def set_six_points(scan: Scan, points: list, *, log: Log = None) -> None:
    """Store the 6 (rx, ry) points picked on the ADF (real-space scan positions)."""
    pts = [(float(x), float(y)) for x, y in points]
    _pipeline().set_bragg_points(scan.ensure_state(), pts, log=log)
    scan.params.six_points = pts


def _push_detect_params(scan: Scan) -> None:
    """Push CalibrationParams.detect_* → state.detect_params (incl. CUDA)."""
    st = scan.ensure_state()
    p = scan.params
    st.detect_params = {
        "minAbsoluteIntensity": int(p.detect_min_absolute_intensity),
        "minRelativeIntensity": float(p.detect_min_relative_intensity),
        "minPeakSpacing":       int(p.detect_min_peak_spacing),
        "edgeBoundary":         int(p.detect_edge_boundary),
        "sigma":                float(p.detect_sigma),
        "maxNumPeaks":          int(p.detect_max_num_peaks),
        "subpixel":             str(p.detect_subpixel),
        "corrPower":            float(p.detect_corr_power),
        "CUDA":                 bool(p.detect_cuda),
        "CUDA_batched":         bool(p.detect_cuda),
    }


def detect_preview(scan: Scan, *, make_figure: bool = True, log: Log = None):
    """Preview detection at the 6 points with current detect_params (tuning).

    Captures whatever figure py4DSTEM draws into ``scan.figures['select6']`` so
    the 6-point comparison shows up in the Report.
    """
    _push_detect_params(scan)
    st = scan.ensure_state()
    _sync_six_points(scan, log=log)        # ensure the picked points are in the state
    if make_figure:
        with capture_pyplot_figures() as figs:
            disks = _pipeline().detect_selected_bragg_disks_step(st, log=log)
        if figs:
            register_figure(scan, "select6", figs[-1])
        return disks
    return _pipeline().detect_selected_bragg_disks_step(st, log=log)


def _sync_six_points(scan: Scan, *, log: Log = None) -> None:
    """Push ``scan.params.six_points`` into the WorkflowState if the state doesn't
    already carry 6 points. The GUI stores picks on params; detection reads them off
    the state (and a heavy datacube reset can drop them), so re-sync defensively."""
    st = scan.ensure_state()
    if len(getattr(st, "bragg_rxs", ()) or ()) == 6:
        return
    pts = scan.params.six_points or []
    if len(pts) == 6:
        _pipeline().set_bragg_points(st, [(float(x), float(y)) for x, y in pts], log=log)


# ── Interactive 6-point detection tuner (mirrors the notebook's live cell) ──────
# Display-only view filters (NOT detection params): they only change how the DP
# renders so faint disks become visible. Ported from the notebook's ``view_filter``.
DETECT_VIEW_MODES = [("Raw (normalize)", "raw"), ("Percentile + gamma", "pclip_gamma"),
                     ("Log (log1p)", "log"), ("High-pass (bg subtract)", "highpass")]
DETECT_CMAPS = ["inferno", "magma", "viridis", "plasma", "gray", "cividis", "turbo"]
# six distinct, high-contrast border/point colors (one per picked point)
SIX_POINT_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def _view_filter(dp, mode="highpass", p_lo=1.0, p_hi=99.8, gamma=0.45, hp_sigma=6.0):
    """Display normalization for a diffraction pattern (notebook ``view_filter``).

    Returns a float32 image in [0, 1]. ``highpass`` (default) subtracts a Gaussian
    background so weak Bragg disks pop; it falls back to percentile+gamma if SciPy
    is missing. This NEVER touches detection — it only changes how the DP looks."""
    x = np.asarray(dp, dtype=np.float32)
    p_lo_ = float(np.clip(p_lo, 0.0, 100.0))
    p_hi_ = float(np.clip(p_hi, 0.0, 100.0))
    if p_hi_ <= p_lo_:
        p_hi_ = min(100.0, p_lo_ + 0.1)
    if mode == "pclip_gamma":
        lo, hi = np.percentile(x, [p_lo_, p_hi_])
        x = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
        return x ** float(max(gamma, 1e-6))
    if mode == "log":
        x = np.log1p(np.clip(x, 0, None))
        hi = np.percentile(x, p_hi_)
        return np.clip(x / (hi + 1e-12), 0, 1)
    if mode == "highpass":
        try:
            from scipy.ndimage import gaussian_filter
            if float(hp_sigma) > 0:
                x = x - gaussian_filter(x, float(hp_sigma))
            x = np.clip(x, 0, None)
            hi = np.percentile(x, p_hi_)
            return np.clip(x / (hi + 1e-12), 0, 1)
        except Exception:
            lo, hi = np.percentile(x, [p_lo_, p_hi_])
            return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
    # raw
    hi = np.percentile(x, p_hi_)
    return np.clip(x / (hi + 1e-12), 0, 1)


def detect_six_points(scan: Scan, *, log: Log = None):
    """Run find_Bragg_disks at the 6 picked points with the scan's detect_params.

    Pushes detect_params + syncs the 6 points first, then calls the proven
    ``detect_selected_bragg_disks_step`` (needs datacube + probe + 6 points).
    Returns the per-point detected-disks list (state.selected_disks)."""
    _push_detect_params(scan)
    _sync_six_points(scan, log=log)
    return _pipeline().detect_selected_bragg_disks_step(scan.ensure_state(), log=log)


def _dp_at(dc, ry, rx):
    """A single diffraction pattern at real-space (ry, rx) from the datacube."""
    try:
        return np.asarray(dc.data[int(ry), int(rx)], dtype=np.float32)
    except Exception:
        return np.asarray(dc[int(ry), int(rx), :, :], dtype=np.float32)


def build_six_point_detection_figure(scan: Scan, *, view_mode: str = "highpass",
                                     cmap: str = "inferno", log: Log = None,
                                     p_lo: float = 1.0, p_hi: float = 99.8,
                                     gamma: float = 0.45, hp_sigma: float = 6.0):
    """Live 2x3 grid of the 6 DPs with detected Bragg disks overlaid (open circles).

    The GUI's interactive detection tuner: edit detect_params → this re-runs
    detection on the 6 points and redraws. Mirrors the notebook's show_image_grid +
    view_filter. Requires datacube + probe + 6 points (raises a clear error if not).
    Returns a matplotlib Figure; also registers it as ``scan.figures['select6']``.
    """
    from matplotlib.figure import Figure
    st = scan.ensure_state()
    disks = detect_six_points(scan, log=log)
    dc = st.datacube
    rxs = list(getattr(st, "bragg_rxs", ()) or [])
    rys = list(getattr(st, "bragg_rys", ()) or [])
    n = min(6, len(rxs), len(rys))
    fig = Figure(figsize=(12.2, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 5)
    # LEFT: ADF (real space) with the 6 picked points in their matching colors
    ax_adf = fig.add_subplot(gs[:, 0:2])
    adf = cached_adf(scan)
    if adf is not None and getattr(adf, "ndim", 0) == 2:
        v = adf[adf > 0]
        vmin, vmax = ((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                      if v.size else (float(adf.min()), float(adf.max() or 1)))
        ax_adf.imshow(adf, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        for i in range(n):
            c = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
            ax_adf.scatter([rxs[i]], [rys[i]], s=90, facecolors="none",
                           edgecolors=c, linewidths=2.2)
            ax_adf.text(rxs[i] + 3, rys[i] - 3, str(i + 1), color=c, fontsize=9,
                        fontweight="bold")
        ax_adf.set_title("ADF — 6 selected points", fontsize=9)
    else:
        ax_adf.text(0.5, 0.5, "no ADF (load the .h5)", ha="center", va="center")
    ax_adf.set_xticks([]); ax_adf.set_yticks([])
    # RIGHT: the 6 DPs with detected disks (open circles), colored borders
    for i in range(n):
        r, col = divmod(i, 3)
        ax = fig.add_subplot(gs[r, 2 + col])
        try:
            img = _view_filter(_dp_at(dc, rys[i], rxs[i]), mode=view_mode,
                              p_lo=p_lo, p_hi=p_hi, gamma=gamma, hp_sigma=hp_sigma)
            ax.imshow(img, cmap=cmap, origin="upper")
        except Exception as exc:
            ax.text(0.5, 0.5, f"DP error:\n{exc}", ha="center", va="center", fontsize=7)
        color = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
        npk = 0
        try:                                     # overlay detected disks (open circles)
            d = disks[i].data
            ax.scatter(d["qy"], d["qx"], s=140, facecolors="none",
                       edgecolors=color, linewidths=1.3)
            npk = len(d)
        except Exception:
            pass
        for sp in ax.spines.values():
            sp.set_color(color); sp.set_linewidth(2.2)
        ax.set_title(f"pt{i+1}  rx={rxs[i]} ry={rys[i]}  ·  {npk} disks", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    register_figure(scan, "select6", fig)
    return fig


def compute_braggpeaks(scan: Scan, save_path: str | None = None,
                       *, log: Log = None) -> None:
    """Full-scan disk detection → braggpeaks (cells 16-18). HEAVY (CUDA).

    Pushes the tuned detect_params first, runs find_Bragg_disks over the whole
    scan, and (optionally) saves braggpeaks.h5 — the file that feeds calibration.
    """
    st = scan.ensure_state()
    _push_detect_params(scan)
    _pipeline().compute_braggpeaks_step(st, save_path=save_path, log=log)
    if save_path:
        scan.braggpeaks_path = str(save_path)
        st.braggpeaks_path = str(save_path)
    _log(log, f"[{scan.name}] braggpeaks computed (CUDA={scan.params.detect_cuda})")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration  — apply all lightweight calibration params at once.
# Reuses fast_batch._apply_calibration, which is itself grounded in the notebook
# (set_roi_from_bounds → set_origin_center_guess → set_q_pixel_size_step →
#  update_strain_basis_params → setup_basis_step → update_strain_params).
# ─────────────────────────────────────────────────────────────────────────────

def _params_to_batch_scan_cfg(scan: Scan):
    """Translate CalibrationParams → fast_batch.FastBatchScanConfig."""
    fb = _batch()
    p = scan.params
    return fb.FastBatchScanConfig(
        name=scan.name,
        raw_path=scan.raw_path,
        braggpeaks_path=scan.braggpeaks_path,
        roi_bounds=list(p.roi_bounds) if p.roi_bounds else [],
        center_guess=list(p.center_guess),
        step10={"sampling": int(p.origin_sampling)},
        step11={
            "px_guess": float(p.q_px),
            "kmax":     float(p.q_kmax),
            "kpow":     float(p.q_kpow),
            "use_roi":  bool(p.q_use_roi),
        },
        step12={
            "qr_rotation":    float(p.qr_rotation),
            "qr_flip":        bool(p.qr_flip),
            "manual_enabled": bool(p.basis_manual_enabled),
            "choose_basis_vectors": {
                "minSpacing":           int(p.min_spacing),
                "minAbsoluteIntensity": int(p.min_absolute_intensity),
                "maxNumPeaks":          int(p.max_num_peaks),
                "edgeBoundary":         int(p.edge_boundary),
                "index_origin":         int(p.index_origin),
                "index_g1":             int(p.index_g1),
                "index_g2":             int(p.index_g2),
                "vis_params":           {"vmin": float(p.vis_vmin),
                                          "vmax": float(p.vis_vmax)},
            },
        },
        step13={
            "coordinate_rotation": float(p.coordinate_rotation),
            "max_peak_spacing":    float(p.max_peak_spacing),
            "vrange":              list(p.vrange),
            "vrange_theta":        list(p.vrange_theta),
            "layout":              str(p.strain_layout),
            "cmap":                str(p.strain_cmap),
            "cmap_theta":          str(p.strain_cmap_theta),
            "show_orientation":    bool(p.strain_show_orientation),
            "scan_roi_bounds":     p.strain_scan_roi_bounds,
        },
        stress_cfg={},
        line_positions={},
        line_width=1,
    )


def apply_calibration(scan: Scan, *, log: Log = None) -> None:
    """Bulk-apply pre-fitted calibration *values* (no refit, no ellipse).

    Fast path for re-applying known params to a fresh state (e.g. template /
    multi where braggpeaks already carry origin+ellipse). For a from-scratch
    calibration that runs each fit in order, use ``run_calibration_sequence``.
    """
    st = scan.ensure_state()
    sc_cfg = _params_to_batch_scan_cfg(scan)
    _batch()._apply_calibration(st, sc_cfg, log)
    _log(log, f"[{scan.name}] calibration values applied (bulk)")


def _build_crystal(scan: Scan):
    """Build a py4DSTEM Crystal from the scan's cal_crystal (Si/Au/Custom/CIF)."""
    import py4DSTEM
    p = scan.params
    if p.cal_crystal == "CIF" and p.cif_path:
        cif = Path(p.cif_path)
        if not cif.is_file():
            raise FileNotFoundError(f"CIF not found: {cif}")
        return py4DSTEM.process.diffraction.Crystal.from_CIF(
            str(cif),
            primitive=False,
            conventional_standard_structure=True,
        )
    cc = p.cal_crystal_obj()
    if p.cal_crystal in ("Si", "Au"):
        # reuse the proven preset builder (Si=diamond, Au=fcc)
        return _pipeline()._make_crystal(p.cal_crystal)
    # Custom: build directly from the edited a_lat / atom_num / positions.
    # atom_num may be a single Z (all sites) or a per-site list (e.g. zincblende GaAs).
    positions = np.asarray(cc.positions, dtype=float)
    numbers = (np.asarray(cc.atom_num, dtype=int) if isinstance(cc.atom_num, (list, tuple))
               else int(cc.atom_num))
    return py4DSTEM.process.diffraction.Crystal(positions, numbers, float(cc.a_lat))


# ─────────────────────────────────────────────────────────────────────────────
# Ordered calibration steps (the canonical workflow, grounded in the notebook).
#
#   ROI → ORIGIN → ELLIPSE(off) → Q-PIXEL → BASIS → (STRAIN)
#
#   • ROI    : ONE region on the ADF, set up front; shared by ellipse/q-pixel.
#   • ORIGIN : uses center_guess "pickup" (NOT the ROI): measure_origin+fit_origin.
#   • ELLIPSE: optional (off by default); fit_ellipse_1D over the shared ROI.
#   • Q-PIXEL: always fits — px_guess in → fitted px out (crystal Si/Au/Custom).
#   • BASIS  : choose_basis_vectors (+ QR rotation/flip/manual indices).
#
# Each step optionally produces ONE figure into ``scan.figures[step]`` for the
# multi-scan Report (all calibrations + maps per scan).
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATION_ORDER = ("roi", "origin", "ellipse", "q_pixel", "basis")


def _store_fig(scan: Scan, step: str, fig) -> None:
    register_figure(scan, step, fig)


def _real_space_shape(st) -> tuple[int, int] | None:
    """Real-space (Ry, Rx) shape from datacube → ADF → braggpeaks (Path A safe)."""
    dc = getattr(st, "datacube", None)
    if dc is not None and getattr(dc, "Rshape", None) is not None:
        return tuple(int(v) for v in dc.Rshape)
    vi = getattr(st, "virtual_images", None) or {}
    adf = vi.get("annular_dark_field")
    if adf is not None:
        arr = getattr(adf, "data", adf)
        if getattr(arr, "ndim", 0) == 2:
            return (int(arr.shape[0]), int(arr.shape[1]))
    bp = getattr(st, "braggpeaks", None)
    rs = getattr(bp, "Rshape", None)
    if rs is not None and len(rs) == 2:
        return (int(rs[0]), int(rs[1]))
    return None


# ── 1. ROI ────────────────────────────────────────────────────────────────────
def set_roi(scan: Scan, *, log: Log = None) -> None:
    """Apply the single ADF ROI shared by all later calibration steps.

    The proven ``set_roi_from_bounds`` needs the full datacube (it reads
    ``datacube.Rshape``). In Path A we only load braggpeaks (+ ADF sidecar), so
    we build the same ``roi_bounds`` / ``roi_mask`` from whatever real-space
    shape is available — no heavy datacube load required.
    """
    st = scan.ensure_state()
    if not scan.params.roi_bounds:
        return
    if getattr(st, "datacube", None) is not None:
        _pipeline().set_roi_from_bounds(st, tuple(scan.params.roi_bounds), log)
    else:
        import numpy as np
        shape = _real_space_shape(st)
        if shape is None:
            raise RuntimeError("Cannot set ROI: no datacube, ADF, or braggpeaks "
                               "real-space shape available for this scan.")
        height, width = shape
        x0, x1, y0, y1 = (int(v) for v in scan.params.roi_bounds)
        x0 = max(0, min(x0, width - 1)); x1 = max(1, min(x1, width))
        y0 = max(0, min(y0, height - 1)); y1 = max(1, min(y1, height))
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Invalid ROI bounds; require x0 < x1 and y0 < y1.")
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        st.roi_bounds = (x0, x1, y0, y1)
        st.roi_mask = mask
        _log(log, f"[{scan.name}] ROI mask built from {height}x{width} (no datacube)")
    _log(log, f"[{scan.name}] ROI set {scan.params.roi_bounds}")


# ── 2. ORIGIN ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION CHECKPOINTS — in-memory snapshots of braggpeaks.calibration so that
# re-testing / re-applying a calibration starts from the clean PRE-step state
# (origin → ellipse → q-pixel → basis are cumulative; without this an ellipse
# applied twice would compound on the already-calibrated state).
# ─────────────────────────────────────────────────────────────────────────────

# canonical cumulative order of the interactive calibrations
CHECKPOINT_ORDER = ["origin", "ellipse", "qpixel", "basis"]


def _scan_calibration(scan: Scan):
    """The live ``braggpeaks.calibration`` for this scan (or None)."""
    st = getattr(scan, "state", None)
    bp = getattr(st, "braggpeaks", None) if st is not None else None
    return getattr(bp, "calibration", None) if bp is not None else None


def snapshot_calibration(scan: Scan):
    """A deep, in-memory copy of the current calibration state (fast, ~MB). Prefers
    the emdfile ``_params`` dict; falls back to copying the whole object."""
    import copy
    cal = _scan_calibration(scan)
    if cal is None:
        return None
    params = getattr(cal, "_params", None)
    try:
        if isinstance(params, dict):
            return {"params": copy.deepcopy(params)}
    except Exception:
        pass
    try:
        return {"obj": copy.deepcopy(cal)}
    except Exception:
        return None


def restore_calibration(scan: Scan, snap, *, log: Log = None) -> bool:
    """Restore a snapshot from ``snapshot_calibration`` and re-run ``setcal()``."""
    import copy
    if not snap:
        return False
    st = getattr(scan, "state", None)
    bp = getattr(st, "braggpeaks", None) if st is not None else None
    cal = getattr(bp, "calibration", None) if bp is not None else None
    if cal is None:
        return False
    try:
        if "params" in snap and isinstance(getattr(cal, "_params", None), dict):
            cal._params.clear()
            cal._params.update(copy.deepcopy(snap["params"]))
        elif "obj" in snap:
            bp.calibration = copy.deepcopy(snap["obj"])
        else:
            return False
        try:
            bp.setcal()
        except Exception:
            pass
        return True
    except Exception as exc:
        _log(log, f"[{scan.name}] calibration restore failed: {exc}")
        return False


def ensure_pre_step_checkpoint(scan: Scan, step: str, *, log: Log = None) -> None:
    """Make the calibration baseline for ``step`` (origin/ellipse/qpixel/basis) the
    clean PRE-step state, then drop now-stale downstream baselines.

    First time the step is applied → snapshot the current (pre-step) state. On every
    later apply of the same step → RESTORE that snapshot first, so the new fit/apply
    runs on the same clean baseline instead of compounding on the previous result.
    """
    if step not in CHECKPOINT_ORDER:
        return
    cps = scan.cal_checkpoints
    key = f"pre_{step}"
    if key in cps and cps[key] is not None:
        if restore_calibration(scan, cps[key], log=log):
            _log(log, f"[{scan.name}] reset to '{key}' baseline (no compounding).")
    else:
        snap = snapshot_calibration(scan)
        cps[key] = snap
        _log(log, f"[{scan.name}] saved '{key}' baseline"
                  + ("" if snap else " (snapshot unavailable — reset disabled)"))
    # downstream baselines were captured on a now-superseded state → invalidate
    for later in CHECKPOINT_ORDER[CHECKPOINT_ORDER.index(step) + 1:]:
        cps.pop(f"pre_{later}", None)


def reset_to_pre_step(scan: Scan, step: str, *, log: Log = None) -> bool:
    """Manually revert to the PRE-step baseline (un-apply this step + everything
    downstream). Returns True if a baseline existed and was restored."""
    cps = getattr(scan, "cal_checkpoints", {})
    snap = cps.get(f"pre_{step}")
    if not snap:
        _log(log, f"[{scan.name}] no '{step}' baseline to reset to "
                  f"(apply '{step}' once first).")
        return False
    ok = restore_calibration(scan, snap, log=log)
    if ok:
        for later in CHECKPOINT_ORDER[CHECKPOINT_ORDER.index(step) + 1:]:
            cps.pop(f"pre_{later}", None)
        _log(log, f"[{scan.name}] reverted to PRE-{step} state.")
    return ok


def save_checkpoint_h5(scan: Scan, stage: str, out_dir: str | None = None,
                       *, log: Log = None) -> str:
    """Durable checkpoint: write the current braggpeaks (with its calibration) to an
    h5 — the notebook's ``py4DSTEM.save(..., mode='o')`` pattern. Optional/explicit;
    the automatic reset uses the in-memory snapshot above."""
    import py4DSTEM
    st = getattr(scan, "state", None)
    bp = getattr(st, "braggpeaks", None) if st is not None else None
    if bp is None:
        raise RuntimeError("No braggpeaks loaded — nothing to checkpoint.")
    base = out_dir or (str(Path(scan.braggpeaks_path).parent) if scan.braggpeaks_path
                       else str(Path(tempfile.gettempdir()) / "fast4d_checkpoints"))
    Path(base).mkdir(parents=True, exist_ok=True)
    path = str(Path(base) / f"{scan.name}_checkpoint_{stage}.h5")
    _log(log, f"[{scan.name}] saving checkpoint → {path}")
    py4DSTEM.save(path, bp, mode="o")
    _log(log, f"[{scan.name}] checkpoint saved ({stage}).")
    return path


def calibrate_origin(scan: Scan, *, make_figure: bool = True, log: Log = None) -> None:
    """Origin correction: measure_origin(center_guess) → fit_origin → setcal."""
    pl = _pipeline()
    st = scan.ensure_state()
    p = scan.params
    pl.set_origin_center_guess(st, tuple(p.center_guess),
                               sampling=int(p.origin_sampling), log=log)
    result = pl.run_origin_correction_step(st, sampling=int(p.origin_sampling), log=log)
    if make_figure:
        try:
            _store_fig(scan, "origin",
                       pl.build_origin_correction_result_figure(st, result, log=log))
        except Exception as exc:
            _log(log, f"[{scan.name}] origin figure skipped: {exc}")
    _log(log, f"[{scan.name}] origin corrected")


# ── 3. ELLIPSE (optional) ─────────────────────────────────────────────────────
def calibrate_ellipse(scan: Scan, *, make_figure: bool = True, log: Log = None) -> None:
    """Ellipse calibration (OFF unless params.ellipse_enabled). Uses shared ROI."""
    p = scan.params
    if not p.ellipse_enabled:
        try:
            _pipeline().mark_ellipse_calibration_skipped(scan.ensure_state(), log=log)
        except Exception:
            pass
        _log(log, f"[{scan.name}] ellipse skipped")
        return
    pl = _pipeline()
    st = scan.ensure_state()
    r0, r1 = sorted(int(v) for v in p.ellipse_q_range)
    center = (tuple(map(float, p.ellipse_center))
              if getattr(p, "ellipse_center", None) else None)
    res = pl.fit_ellipse_step(st, (r0, r1), sampling=int(p.ellipse_sampling),
                              use_roi=bool(p.ellipse_use_roi),
                              center=center, log=log)
    pl.apply_ellipse_step(st, log=log)
    if make_figure:
        try:
            _store_fig(scan, "ellipse", _ellipse_figure(res, st))
        except Exception as exc:
            _log(log, f"[{scan.name}] ellipse figure skipped: {exc}")
    _log(log, f"[{scan.name}] ellipse fitted+applied (q_range={r0},{r1})")


def ellipse_bvm(scan: Scan, *, sampling: int = 1, use_roi: bool = False, log: Log = None):
    """Calibrated Bragg vector map for the interactive ellipse tool.

    Returns ``(img2D, origin(y0,x0), bvm)``. Uses ``histogram(mode='cal')`` (centred
    on the calibrated origin) — falls back to ``mode='raw'`` if calibration isn't set.
    """
    pl = _pipeline()
    st = scan.ensure_state()
    if getattr(st, "braggpeaks", None) is None:
        load_braggpeaks(scan, log=log)
    bp = pl._braggpeaks_for_roi(st, use_roi=bool(use_roi))
    try:
        bvm = bp.histogram(mode="cal", sampling=int(sampling))
    except Exception:
        bvm = bp.histogram(mode="raw", sampling=int(sampling))
    img = np.asarray(getattr(bvm, "data", bvm), dtype=float)
    origin = getattr(bvm, "origin", None)
    origin = (tuple(map(float, origin)) if origin is not None
              else (img.shape[0] / 2.0, img.shape[1] / 2.0))
    return img, origin, bvm


def fit_ellipse_preview(scan: Scan, r0, r1, *, sampling: int = 1,
                        use_roi: bool = False, bvm=None,
                        center: tuple[float, float] | None = None,
                        log: Log = None) -> dict:
    """Fit the ellipse over the ring (r0,r1) WITHOUT applying (notebook fit_ellipse_1D).

    Pass a cached ``bvm`` from a recent :func:`ellipse_bvm` call to skip recomputing the
    histogram (the expensive step). ``center``, if given, is a ``(y, x)`` BVM-pixel point
    used as the annulus/initial-guess center for the fit instead of ``bvm.origin`` (the
    calibrated probe origin) — lets the fit be steered toward the visual BVM ring center.
    Returns ``{ok, img, origin, p_ellipse(y0,x0,a,b,theta),
    a, b, theta, ab, shift, ok_scale, R, r0, r1}``. ``ok=False`` if the fit returns nothing.
    """
    import py4DSTEM
    if bvm is None:
        img, origin, bvm = ellipse_bvm(scan, sampling=sampling, use_roi=use_roi, log=log)
    else:
        img = np.asarray(getattr(bvm, "data", bvm), dtype=float)
        origin = getattr(bvm, "origin", None)
        origin = (tuple(map(float, origin)) if origin is not None
                  else (img.shape[0] / 2.0, img.shape[1] / 2.0))
    r0, r1 = sorted((int(r0), int(r1)))
    fit_center = tuple(map(float, center)) if center is not None else bvm.origin
    try:
        p_fit = py4DSTEM.process.calibration.fit_ellipse_1D(
            bvm, center=fit_center, fitradii=(r0, r1))
    except Exception as exc:
        _log(log, f"[{scan.name}] ellipse fit failed: {exc}")
        return {"ok": False, "img": img, "origin": origin, "r0": r0, "r1": r1}
    if p_fit is None:
        return {"ok": False, "img": img, "origin": origin, "r0": r0, "r1": r1}
    y0, x0, a, b, theta = map(float, p_fit)
    y0o, x0o = origin
    shift = float(np.hypot(y0 - y0o, x0 - x0o))
    R = 0.5 * (r0 + r1)
    ok_scale = (0.5 * R <= a <= 2.0 * R) and (0.5 * R <= b <= 2.0 * R)
    _log(log, f"[{scan.name}] ellipse fit: a={a:.3f} b={b:.3f} "
              f"theta={np.degrees(theta):.2f}deg shift={shift:.2f}px scale_ok={ok_scale}")
    return {"ok": True, "img": img, "origin": origin, "p_ellipse": (y0, x0, a, b, theta),
            "a": a, "b": b, "theta": theta, "ab": (a / b if b else float("nan")),
            "shift": shift, "ok_scale": ok_scale, "R": R, "r0": r0, "r1": r1}


def apply_ellipse_fit(scan: Scan, p_ellipse, *, r_range=None,
                      make_figure: bool = True, log: Log = None) -> None:
    """Apply a fitted ellipse (caller should reset to the pre-ellipse baseline first).

    Mirrors the notebook: ``calibration.set_p_ellipse(p) → setcal()``. Marks
    ``ellipse_enabled`` and stores the q_range so a later full Compute reproduces it.
    """
    pl = _pipeline()
    st = scan.ensure_state()
    bp = pl.require_braggpeaks(st)
    p = tuple(float(v) for v in p_ellipse)
    bp.calibration.set_p_ellipse(p)
    bp.setcal()
    scan.params.ellipse_enabled = True
    if r_range:
        scan.params.ellipse_q_range = [int(min(r_range)), int(max(r_range))]
    _log(log, f"[{scan.name}] ellipse applied p={tuple(round(v, 3) for v in p)}")
    if make_figure:
        try:
            _store_fig(scan, "ellipse", _ellipse_overlay_figure(scan, p, r_range))
        except Exception as exc:
            _log(log, f"[{scan.name}] ellipse figure skipped: {exc}")


def _ellipse_overlay_figure(scan: Scan, p_ellipse, r_range=None, *,
                            img=None, origin=None):
    """BVM with the fitted ellipse (cyan) + the ring radii (yellow dashed) drawn."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Ellipse as mplEllipse, Circle as mplCircle
    if img is None:
        img, origin, _ = ellipse_bvm(scan)
    y0, x0, a, b, theta = (float(v) for v in p_ellipse)
    fig = Figure(figsize=(4.6, 4.4), constrained_layout=True)
    ax = fig.add_subplot(111)
    v = img[img > 0]
    vmax = float(np.percentile(v, 99)) if v.size else 1.0
    ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax, origin="upper")
    ax.add_patch(mplEllipse((x0, y0), 2 * a, 2 * b, angle=np.degrees(theta),
                            fill=False, edgecolor="cyan", lw=1.6))
    for r in (r_range or []):
        ax.add_patch(mplCircle((x0, y0), float(r), fill=False,
                               edgecolor="yellow", lw=0.8, ls="--"))
    ax.plot([x0], [y0], "r+", ms=10)
    ax.set_title(f"Ellipse  a={a:.2f}  b={b:.2f}  θ={np.degrees(theta):.1f}°  "
                 f"a/b={a / b:.4f}" if b else "Ellipse", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    register_figure(scan, "ellipse", fig, force=True)
    return fig


def _ellipse_figure(res: dict, st):
    """BVM with the FITTED ELLIPSE drawn on top (cyan) + the fit ring (yellow dashed)
    — the compute-time figure was showing only the BVM, not the ellipse."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Ellipse as mplEllipse, Circle as mplCircle
    fig = Figure(figsize=(4.4, 4.2), constrained_layout=True)
    ax = fig.add_subplot(111)
    bvm = res.get("bvm")
    img = np.asarray(getattr(bvm, "data", bvm)) if bvm is not None else None
    if img is not None and img.ndim == 2:
        v = img[img > 0]
        vmax = float(np.percentile(v, 99)) if v.size else 1.0
        ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax, origin="upper")
    p_ell = res.get("p_ellipse")
    if p_ell is not None and len(p_ell) >= 5:
        y0, x0, a, b, theta = (float(t) for t in p_ell[:5])
        ax.add_patch(mplEllipse((x0, y0), 2 * a, 2 * b, angle=np.degrees(theta),
                                fill=False, edgecolor="#00E5FF", lw=1.8))
        for r in (res.get("fitradii") or res.get("q_range") or []):
            ax.add_patch(mplCircle((x0, y0), float(r), fill=False,
                                   edgecolor="yellow", lw=0.8, ls="--"))
        ax.plot([x0], [y0], "r+", ms=9)
        ax.set_title(f"Ellipse fit  a={a:.2f} b={b:.2f} θ={np.degrees(theta):.1f}°  "
                     f"a/b={a / b:.4f}" if b else "Ellipse fit", fontsize=8)
    else:
        ax.set_title("Ellipse (no fit)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    return fig


# ── 4. Q-PIXEL ────────────────────────────────────────────────────────────────
def braggpeaks_has_origin(scan: Scan) -> bool:
    """True when braggpeaks carry a diffraction origin (required for Q-pixel fit)."""
    st = getattr(scan, "state", None)
    bp = getattr(st, "braggpeaks", None) if st is not None else None
    if bp is None:
        return False
    try:
        cal = getattr(bp, "calibration", None)
        if cal is not None and getattr(cal, "get_origin", None) is not None:
            if cal.get_origin() is not None:
                return True
    except Exception:
        pass
    try:
        cs = getattr(bp, "calstate", None)
        if isinstance(cs, dict) and cs.get("center"):
            return True
        if cs is not None and bool(getattr(cs, "get", lambda *_: False)("center", False)):
            return True
        if cs is not None and bool(getattr(cs, "center", False)):
            return True
    except Exception:
        pass
    return False


def ensure_origin_for_qpixel(scan: Scan, *, log: Log = None) -> None:
    """Q-pixel crystal fit always requests ``center=True`` in py4DSTEM.

    If origin is missing, run Origin from ``center_guess`` (same as Compute Calib=fit).
    Raises a clear error when even that cannot proceed.
    """
    st = scan.ensure_state()
    if getattr(st, "braggpeaks", None) is None:
        load_braggpeaks(scan, log=log)
    if braggpeaks_has_origin(scan):
        return
    cg = getattr(scan.params, "center_guess", None)
    if not cg:
        raise RuntimeError(
            f"[{scan.name}] Q-pixel fit needs Origin first — braggpeaks have "
            f"calstate.center=False. Open Origin → set center_guess → Apply calibration "
            f"(or use Setting Q-pixel / Through / Compute with Calib=fit).")
    _log(log, f"[{scan.name}] Origin missing on braggpeaks — running Origin "
              f"(required before Q-pixel fit) from center_guess={list(cg)}…")
    calibrate_origin(scan, make_figure=True, log=log)
    if not braggpeaks_has_origin(scan):
        raise RuntimeError(
            f"[{scan.name}] Origin ran but calstate.center is still False — "
            f"cannot fit Q-pixel. Re-run Origin (Apply calibration) and check the log.")


def calibrate_q_pixel(scan: Scan, *, refit: bool = True,
                      make_figure: bool = True, log: Log = None) -> float:
    """Calibrate the Q pixel size (notebook cell 37).

    refit=True  (default): build the selected crystal (Si/Au/Custom), run
                ``crystal.calibrate_pixel_size`` over braggpeaks (masked to the
                shared ROI when ``q_use_roi``), and store the fitted px in
                ``scan.params.q_px_fitted`` (the guess ``q_px`` is left untouched).
                NOTE: calibrate_pixel_size fits from the structure factors — the
                guess does not seed it, so the fitted value is what gets used.
                Requires a calibrated origin (auto-runs Origin from center_guess
                when missing).
    refit=False : use ``params.q_px`` (the GUESS) directly as the pixel size, no
                fit — pick this to keep your own calibrated value.

    Returns the px value now in effect (Å⁻¹/px).
    """
    pl = _pipeline()
    st = scan.ensure_state()
    p = scan.params
    # propagate the chosen crystal so q_pixel_overlay_figure uses it
    try:
        st.q_crystal = p.cal_crystal if p.cal_crystal in ("Si", "Au") else "Si"
    except Exception:
        pass

    def _fit_threadsafe() -> float:
        """The proven fast-mode fit WITHOUT pyplot (sanity-checked). Returns px_fit."""
        ensure_origin_for_qpixel(scan, log=log)
        crystal = _build_crystal(scan)            # honours Si / Au / Custom
        crystal.calculate_structure_factors(float(p.q_kmax))
        bragg_use = pl._braggpeaks_for_roi(st, use_roi=bool(p.q_use_roi))
        pl._sync_q_pixel_to_objects(st, bragg_use, float(p.q_px))   # seed the guess
        crystal.calibrate_pixel_size(
            bragg_peaks=bragg_use, bragg_k_power=float(p.q_kpow),
            k_max=float(p.q_kmax), set_calibration_in_place=True,
            verbose=False, plot_result=False)
        pxf = float(pl._get_q_pixel_size(bragg_use.calibration))
        if not (0.0 < pxf < 1.0):                 # fast-mode sanity-check
            _log(log, f"[{scan.name}] implausible Q-pixel fit {pxf:.6g} — "
                      f"reverting to the guess {p.q_px:.6g}")
            pxf = float(p.q_px)
        pl._sync_q_pixel_to_objects(st, bragg_use, pxf)
        return pxf

    if not refit:
        pl.set_q_pixel_size_step(st, float(p.q_px), log=log)
        if make_figure:                           # refit OFF → the guess overlay is correct
            try:
                fig = pl.q_pixel_overlay_figure(
                    st, px=float(p.q_px), k_max=float(p.q_kmax),
                    bragg_k_power=float(p.q_kpow), use_roi=bool(p.q_use_roi), log=log)
                _annotate_qpixel_figure(fig, px_applied=float(p.q_px), px_guess=float(p.q_px),
                                        px_fit=None, use_roi=bool(p.q_use_roi),
                                        kpow=float(p.q_kpow), kmax=float(p.q_kmax))
                _store_fig(scan, "q_pixel", fig)
            except Exception as exc:
                _log(log, f"[{scan.name}] q-pixel figure skipped: {exc}")
        return float(p.q_px)

    # refit ON — fit thread-safely; FIT figure needs the GUI thread (pyplot).
    px_fit = _fit_threadsafe()
    p.q_px_fitted = px_fit
    _log(log, f"[{scan.name}] Q-pixel refit ({p.cal_crystal}): guess={p.q_px:.6g} → "
              f"fitted={px_fit:.6g} A^-1/px (use_roi={p.q_use_roi})")
    if make_figure:
        import threading
        if threading.current_thread() is threading.main_thread():
            register_q_pixel_fit_figure(scan, log=log)
        else:
            scan._defer_qpixel_figure = True          # flushed on GUI thread after Compute
    return px_fit


def register_q_pixel_fit_figure(scan: Scan, *, log: Log = None) -> bool:
    """Register the Q-pixel **FIT** figure (py4DSTEM calibrate_pixel_size plot).

    Must run on the **GUI / main thread** — uses pyplot internally. Falls back to a
    scattering overlay at the fitted px if the slow path fails.

    Does **not** reload uncalibrated braggpeaks from disk and re-fit blindly: that
    produced ``Requested calibration was not found!`` after Compute when RAM had
    dropped the calibrated object. Prefer overlay from ``q_px_fitted`` when origin
    is missing after a reload.
    """
    pl = _pipeline()
    st = _set_q_crystal(scan)
    p = scan.params
    px_guess = float(p.q_px)
    px_fit = float(p.q_px_fitted) if p.q_px_fitted not in (None, 0, 0.0) else None

    if getattr(st, "braggpeaks", None) is None:
        # Prefer calibrated in-RAM / checkpoint over a fresh disk reload that
        # wipes Origin (calstate all False) and then fails the FIT re-plot.
        snap = (getattr(scan, "cal_checkpoints", {}) or {}).get("pre_basis") \
            or (getattr(scan, "cal_checkpoints", {}) or {}).get("pre_qpixel")
        restored = False
        if snap:
            try:
                restored = bool(restore_calibration(scan, snap, log=log))
            except Exception:
                restored = False
        if not restored:
            load_braggpeaks(scan, log=log)
            _log(log, f"[{scan.name}] Q-pixel figure: reloaded braggpeaks from disk "
                      f"(uncalibrated). Will restore Origin before any re-fit.")

    try:
        ensure_origin_for_qpixel(scan, log=log)
    except Exception as exc:
        _log(log, f"[{scan.name}] q-pixel FIT figure: cannot ensure Origin ({exc})")
        # Overlay-only fallback at fitted/guess px — never call calibrate_pixel_size
        # without origin (py4DSTEM raises "Requested calibration was not found!").
        try:
            px = float(px_fit if px_fit is not None else px_guess)
            fig = pl.q_pixel_overlay_figure(
                st, px=px, k_max=float(p.q_kmax), bragg_k_power=float(p.q_kpow),
                use_roi=bool(p.q_use_roi), log=log)
            pl._annotate_q_pixel_fit(
                fig, px_guess, px, bool(p.q_use_roi), float(p.q_kpow), float(p.q_kmax))
            register_figure(scan, "q_pixel", fig, force=True)
            _log(log, f"[{scan.name}] registered Q-pixel OVERLAY (no Origin — fit not re-run)")
            return True
        except Exception as exc2:
            _log(log, f"[{scan.name}] q-pixel overlay fallback figure skipped: {exc2}")
            return False

    # Origin OK — rebuild FIT plot. If we already fitted during Compute, still
    # allow py4DSTEM's plot_result path; on failure fall back to overlay.
    try:
        res = pl.finalize_q_pixel_refit_step(
            st, px_guess=px_guess, k_max=float(p.q_kmax),
            bragg_k_power=float(p.q_kpow), use_roi=bool(p.q_use_roi),
            plot_result=True, log=log)
        px_new = float(res.get("px_fit", px_fit or px_guess))
        if 0.0 < px_new < 1.0:
            p.q_px_fitted = px_new
        fig = res.get("figure")
        if fig is not None:
            register_figure(scan, "q_pixel", fig, force=True)
            return True
    except Exception as exc:
        _log(log, f"[{scan.name}] q-pixel FIT figure failed: {exc}")
    try:
        px = float(p.q_px_fitted or px_guess)
        fig = pl.q_pixel_overlay_figure(
            st, px=px, k_max=float(p.q_kmax), bragg_k_power=float(p.q_kpow),
            use_roi=bool(p.q_use_roi), log=log)
        pl._annotate_q_pixel_fit(
            fig, px_guess, px, bool(p.q_use_roi), float(p.q_kpow), float(p.q_kmax))
        register_figure(scan, "q_pixel", fig, force=True)
        return True
    except Exception as exc:
        _log(log, f"[{scan.name}] q-pixel overlay fallback figure skipped: {exc}")
    return False


def flush_deferred_qpixel_figures(scans: list, *, log: Log = None) -> int:
    """Build deferred Q-pixel FIT figures on the GUI thread (after background Compute)."""
    n = 0
    for sc in scans:
        if getattr(sc, "_defer_qpixel_figure", False):
            if register_q_pixel_fit_figure(sc, log=log):
                sc._defer_qpixel_figure = False
                n += 1
    return n


def _set_q_crystal(scan: Scan):
    """Mirror the chosen cal crystal onto state.q_crystal (Si/Au; Custom→Si) so the
    proven pipeline Q-pixel functions (_make_crystal) pick the right reference."""
    st = scan.ensure_state()
    try:
        st.q_crystal = scan.params.cal_crystal if scan.params.cal_crystal in ("Si", "Au") else "Si"
    except Exception:
        pass
    return st


def q_pixel_overlay(scan: Scan, *, px, k_max, kpow, use_roi, log: Log = None):
    """Interactive Q-pixel 'Update': the theory/experimental scattering overlay for the
    current px / k_max / bragg_k_power (delegates to the proven pipeline function)."""
    pl = _pipeline()
    st = _set_q_crystal(scan)
    if getattr(st, "braggpeaks", None) is None:
        load_braggpeaks(scan, log=log)
    return pl.q_pixel_overlay_figure(st, px=float(px), k_max=float(k_max),
                                     bragg_k_power=float(kpow), use_roi=bool(use_roi), log=log)


def q_pixel_test(scan: Scan, *, px0, test_step, n_figures, k_max, kpow, use_roi,
                 log: Log = None) -> dict:
    """Interactive Q-pixel 'Test': 2N+1 refit sensitivity sweep around px0 → summary
    figure (guess-vs-fit + residual). Delegates to the proven pipeline function. Does
    NOT change the applied calibration (it restores px afterwards)."""
    pl = _pipeline()
    st = _set_q_crystal(scan)
    if getattr(st, "braggpeaks", None) is None:
        load_braggpeaks(scan, log=log)
    return pl.test_q_pixel_size_step(
        st, px0=float(px0), test_step=float(test_step), n_figures=int(n_figures),
        k_max=float(k_max), bragg_k_power=float(kpow), use_roi=bool(use_roi), log=log)


def q_pixel_finalize(scan: Scan, *, px_guess, k_max, kpow, use_roi,
                     log: Log = None) -> dict:
    """Interactive Q-pixel 'Finalize / REFIT': the proven pipeline refit (with the
    blue annotation). Stores the fitted px in params.q_px_fitted + registers the
    figure. Caller should reset to the pre-qpixel checkpoint first (no compounding)."""
    pl = _pipeline()
    st = _set_q_crystal(scan)
    if getattr(st, "braggpeaks", None) is None:
        load_braggpeaks(scan, log=log)
    ensure_origin_for_qpixel(scan, log=log)
    res = pl.finalize_q_pixel_refit_step(
        st, px_guess=float(px_guess), k_max=float(k_max), bragg_k_power=float(kpow),
        use_roi=bool(use_roi), plot_result=True, log=log)
    px_fit = float(res.get("px_fit", px_guess))
    if not (0.0 < px_fit < 1.0):                  # same fast-mode sanity-check
        _log(log, f"[{scan.name}] implausible refit {px_fit:.6g} — keeping guess {px_guess:.6g}")
        px_fit = float(px_guess)
        pl._sync_q_pixel_to_objects(st, pl._braggpeaks_for_roi(st, use_roi=bool(use_roi)), px_fit)
    scan.params.q_px_fitted = px_fit
    fig = res.get("figure")
    if fig is not None:
        register_figure(scan, "q_pixel", fig, force=True)
    return {"px_fit": px_fit, "figure": fig}


def _fmt_trim(x, ndp: int = 10) -> str:
    """Trim trailing zeros (notebook ``fmt_trim``)."""
    s = f"{float(x):.{ndp}f}".rstrip("0").rstrip(".")
    return s or "0"


def _annotate_qpixel_figure(fig, *, px_applied, px_guess, px_fit=None,
                            use_roi=False, kpow=2.0, kmax=1.0) -> None:
    """Notebook-style annotation box on the Q-pixel scattering figure (the blue box):
    the guess, the fitted value + delta when a refit ran, else 'applied = guess'.
    Also rewrites the title so it shows the px ACTUALLY applied (= the guess when
    refit is off — your manual calibration, e.g. 0.0137)."""
    if fig is None or not getattr(fig, "axes", None):
        return
    ax = fig.axes[0]
    rows = [f"px_guess = {_fmt_trim(px_guess)} A^-1/px"]
    if px_fit is not None:
        rows.append(f"px_fit   = {_fmt_trim(px_fit)} A^-1/px")
        rows.append(f"delta    = {float(px_fit) - float(px_guess):+.2e}")
        rows.append("APPLIED  = fitted (refit ON)")
    else:
        rows.append("APPLIED  = guess (refit OFF)")
    rows.append(f"ROI={use_roi} | kpow={float(kpow):.2f} | kmax={float(kmax):.2f}")
    try:
        ax.text(0.02, 0.98, "\n".join(rows), transform=ax.transAxes, va="top", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round", facecolor="#BBDEFB",
                          edgecolor="#1565C0", alpha=0.6))
        ax.set_title(f"Q-pixel applied={_fmt_trim(px_applied)} A^-1/px  |  "
                     f"kmax={float(kmax):.2f} | kpow={float(kpow):.2f} | ROI={use_roi}",
                     fontsize=9)
    except Exception:
        pass


# ── 5. BASIS ──────────────────────────────────────────────────────────────────
def calibrate_basis(scan: Scan, *, make_figure: bool = True, log: Log = None) -> None:
    """Basis vectors: choose_basis_vectors (+ QR rotation/flip/manual indices)."""
    pl = _pipeline()
    st = scan.ensure_state()
    p = scan.params
    pl.update_strain_basis_params(
        st,
        min_spacing=int(p.min_spacing),
        min_absolute_intensity=int(p.min_absolute_intensity),
        max_num_peaks=int(p.max_num_peaks),
        edge_boundary=int(p.edge_boundary),
        vmin=float(p.vis_vmin), vmax=float(p.vis_vmax),
        qr_rotation=float(p.qr_rotation), qr_flip=bool(p.qr_flip),
        manual_enabled=bool(p.basis_manual_enabled),
        index_origin=int(p.index_origin), index_g1=int(p.index_g1),
        index_g2=int(p.index_g2), log=log)
    pl.setup_basis_step(st, log)
    _record_basis_g1g2(scan, st, log=log)      # so the cal-state flips to APPLIED
    if make_figure:
        try:
            pl.preview_basis_figure_step(st, log=log, scan_label=scan.name)
            figs = getattr(st, "basis_preview_figures", None) or []
            if figs:
                _store_fig(scan, "basis", figs[0])
        except Exception as exc:
            _log(log, f"[{scan.name}] basis figure skipped: {exc}")
    _log(log, f"[{scan.name}] basis vectors chosen")


def _record_basis_g1g2(scan: Scan, st, *, log: Log = None) -> None:
    """After choose_basis_vectors, copy the chosen reference g1/g2 into
    ``state.strain_basis_params['g1_qxy'|'g2_qxy']`` so pipeline's
    ``_bragg_has_basis_calibration`` (and thus the cal-state strip) reports the basis
    as APPLIED — setup_basis_step alone leaves it merely 'staged'."""
    sm = getattr(st, "strainmap_full", None)
    if sm is None or not isinstance(getattr(st, "strain_basis_params", None), dict):
        return

    def _vec(*attrs):
        for a in attrs:
            v = getattr(sm, a, None)
            if v is not None:
                try:
                    arr = np.asarray(v, dtype=float).ravel()
                    if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                        return [float(arr[0]), float(arr[1])]
                except Exception:
                    pass
        return None

    g1 = _vec("g1", "g1_exp", "g1_meas")
    g2 = _vec("g2", "g2_exp", "g2_meas")
    if g1 is not None and g2 is not None:
        st.strain_basis_params["g1_qxy"] = g1
        st.strain_basis_params["g2_qxy"] = g2
        _log(log, f"[{scan.name}] basis applied (g1={g1}, g2={g2})")
    else:
        _log(log, f"[{scan.name}] basis chosen but g1/g2 not readable from the "
                  f"StrainMap — cal-state may stay 'staged'.")


def basis_preview(scan: Scan, *, log: Log = None):
    """Run choose_basis_vectors with the CURRENT params and return the preview figure
    (for the interactive Basis tool). FAST: runs choose_basis_vectors ONCE (only
    preview_basis_figure_step) — calibrate_basis ran it TWICE (setup + preview), which
    made the live tuner sluggish. Commits the basis to state.strainmap_full (not
    cumulative). Touches pyplot → GUI/main thread only."""
    pl = _pipeline()
    st = scan.ensure_state()
    if getattr(st, "braggpeaks", None) is None:
        load_braggpeaks(scan, log=log)
    p = scan.params
    pl.update_strain_basis_params(
        st, min_spacing=int(p.min_spacing),
        min_absolute_intensity=int(p.min_absolute_intensity),
        max_num_peaks=int(p.max_num_peaks), edge_boundary=int(p.edge_boundary),
        vmin=float(p.vis_vmin), vmax=float(p.vis_vmax),
        qr_rotation=float(p.qr_rotation), qr_flip=bool(p.qr_flip),
        manual_enabled=bool(p.basis_manual_enabled),
        index_origin=int(p.index_origin), index_g1=int(p.index_g1),
        index_g2=int(p.index_g2), log=log)
    fig = None
    try:
        pl.preview_basis_figure_step(st, log=log, scan_label=scan.name)  # 1× choose_basis_vectors
        figs = getattr(st, "basis_preview_figures", None) or []
        if figs:
            fig = figs[0]
    except Exception as exc:
        _log(log, f"[{scan.name}] basis preview skipped: {exc}")
    _record_basis_g1g2(scan, st, log=log)          # cal-state → APPLIED
    return fig



def run_calibration_sequence(scan: Scan, *, make_figures: bool = True,
                             log: Log = None, progress_step=None) -> None:
    """Run the full calibration in canonical order: ROI→origin→ellipse→q-pixel→basis.

    Also sets the strain params (max_peak_spacing, coordinate_rotation, vrange)
    so ``compute_strain`` can run immediately afterwards.
    """
    p = scan.params
    set_roi(scan, log=log)
    if progress_step:
        progress_step("roi")
    calibrate_origin(scan, make_figure=make_figures, log=log)
    if progress_step:
        progress_step("origin")
    calibrate_ellipse(scan, make_figure=make_figures, log=log)
    if progress_step:
        progress_step("ellipse")
    calibrate_q_pixel(scan, refit=bool(p.q_refit), make_figure=make_figures, log=log)
    if progress_step:
        progress_step("q_pixel")
    calibrate_basis(scan, make_figure=make_figures, log=log)
    if progress_step:
        progress_step("basis")
    # strain params (applied at strain time in the notebook, cell 41)
    _pipeline().update_strain_params(
        scan.ensure_state(),
        coordinate_rotation=float(p.coordinate_rotation),
        max_peak_spacing=float(p.max_peak_spacing),
        layout=str(p.strain_layout),
        vrange=list(p.vrange), vrange_theta=list(p.vrange_theta),
        cmap=str(p.strain_cmap), cmap_theta=str(p.strain_cmap_theta),
        show_orientation=bool(p.strain_show_orientation), log=log)
    if p.strain_scan_roi_bounds:
        try:
            _pipeline().set_strain_scan_roi_from_bounds(
                scan.ensure_state(), p.strain_scan_roi_bounds, log)
        except Exception:
            pass
    _log(log, f"[{scan.name}] calibration sequence complete")


def apply_calibrations_through(scan: Scan, upto: str, *, inclusive: bool = False,
                               log: Log = None) -> list:
    """Apply the calibration chain on this scan FROM SCRATCH (origin → ellipse →
    q-pixel → basis), up to ``upto``.

    ``inclusive=False`` → every step BEFORE ``upto`` ('Apply previous calibrations':
    prep the selected file before you tune ``upto``). ``inclusive=True`` → through AND
    INCLUDING ``upto`` ('Apply this calibration to this file': run the whole chain so
    e.g. Basis = origin+ellipse+q-pixel+basis, Q-pixel = origin+ellipse+q-pixel).

    Loads braggpeaks (light) + sets the ROI first; calibrate_ellipse respects
    params.ellipse_enabled (skips when OFF). Returns the steps applied."""
    st = scan.ensure_state()
    if getattr(st, "braggpeaks", None) is None:
        _log(log, f"[{scan.name}] loading braggpeaks…")
        load_braggpeaks(scan, log=log)
    _log(log, f"[{scan.name}] applying ROI from parameter table…")
    set_roi(scan, log=log)
    if upto not in CHECKPOINT_ORDER:
        return []
    end = CHECKPOINT_ORDER.index(upto) + (1 if inclusive else 0)
    done = []
    for step in CHECKPOINT_ORDER[:end]:
        _log(log, f"[{scan.name}] applying calibration step: {step}…")
        if step == "origin":
            calibrate_origin(scan, log=log)
        elif step == "ellipse":
            calibrate_ellipse(scan, log=log)        # no-op when ellipse_enabled is False
        elif step == "qpixel":
            calibrate_q_pixel(scan, refit=bool(scan.params.q_refit), log=log)
        elif step == "basis":
            calibrate_basis(scan, log=log)
        done.append(step)
        _log(log, f"[{scan.name}]   ✓ {step}")
    kind = "through+incl" if inclusive else "previous"
    _log(log, f"[{scan.name}] applied {kind} calibration(s) for '{upto}': {done or 'none'}")
    return done


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE — heavy strain step  (compute_braggpeaks lives in the DETECTION section)
# ─────────────────────────────────────────────────────────────────────────────

def _reference_roi_mask(scan: Scan, rshape) -> "np.ndarray":
    """Boolean real-space mask (shape == StrainMap.rshape) for the with-ROI
    reference region: the strain ROI if set, else the shared calibration ROI."""
    p = scan.params
    bounds = p.strain_scan_roi_bounds or p.roi_bounds
    if not bounds:
        raise RuntimeError("No ROI set — define the ROI (or strain ROI) before the "
                           "with-ROI strain; it provides the reference g1,g2.")
    rs = tuple(int(v) for v in rshape)
    mask = np.zeros(rs, dtype=bool)
    x0, x1, y0, y1 = (int(v) for v in bounds)
    h, w = rs
    x0 = max(0, min(x0, w)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h)); y1 = max(0, min(y1, h))
    mask[y0:y1, x0:x1] = True
    if not mask.any():
        raise RuntimeError(f"With-ROI reference mask is empty for bounds {bounds}.")
    return mask


def compute_strain(scan: Scan, *, use_roi: bool = False,
                   log: Log = None) -> dict:
    """Run the full strain map (cell 41). HEAVY (py4DSTEM get_strain).

    WITHOUT ROI: internal/global reference (the median g1,g2 over the whole scan).
    WITH ROI: the ROI marks the *reference* region — its median g1,g2 become the
    zero-strain basis (``gvects``), so strain is measured relative to that region
    (to compare against the global map). This uses the SAME calibrated braggpeaks
    (origin/ellipse/q-pixel/basis from the table), NOT a masked copy — so the
    with-ROI map honours every calibration the user set. Requires the without-ROI
    StrainMap (it reads the reference g1,g2 off it).

    Stores result figures on ``state.strain_figures`` and raw tensors on
    ``state.strain_raw[label]``. Returns the pipeline result dict.
    """
    st = scan.ensure_state()
    pl = _pipeline()
    p = scan.params
    # Always push table params into state before get_strain (vrange_theta etc.).
    pl.update_strain_params(
        st,
        coordinate_rotation=float(p.coordinate_rotation),
        max_peak_spacing=float(p.max_peak_spacing),
        layout=str(p.strain_layout),
        vrange=list(p.vrange), vrange_theta=list(p.vrange_theta),
        cmap=str(p.strain_cmap), cmap_theta=str(p.strain_cmap_theta),
        show_orientation=bool(p.strain_show_orientation), log=log)
    label = "with_roi" if use_roi else "without_roi"

    gvects = None
    if use_roi:
        sm = getattr(st, "strainmap_full", None)
        if sm is None:                       # need the global map for the reference
            _log(log, f"[{scan.name}] with-ROI needs the global StrainMap first; "
                      f"computing without-ROI…")
            compute_strain(scan, use_roi=False, log=log)
            sm = getattr(st, "strainmap_full", None)
        if sm is None:
            raise RuntimeError("Could not build the global StrainMap for the with-ROI reference.")
        rshape = getattr(sm, "rshape", None)
        if rshape is None:
            raise RuntimeError("StrainMap has no rshape; cannot build the reference ROI.")
        mask = _reference_roi_mask(scan, rshape)
        g1, g2 = pl.capture_reference_g12_from_strainmap(sm, mask, log=log)
        gvects = (g1, g2)

    # use_roi=False here on purpose: we pass the reference via gvects, keeping the
    # FULL calibrated braggpeaks (not a masked, default-calibration copy).
    result = pl.compute_strain_map_step(
        st, use_roi=False, gvects=gvects, label_override=label, log=log)

    # keep figures for later save
    if not getattr(st, "strain_figures", None):
        st.strain_figures = {}
    figs = result.get("figures") or ([result["figure"]] if result.get("figure") else [])
    for i, fig in enumerate(figs):
        st.strain_figures[f"strain_{label}_{i}"] = fig
    # primary strain figure also goes into scan.figures for the Report
    if figs:
        _store_fig(scan, f"strain_{label}", figs[-1])
    _log(log, f"[{scan.name}] strain computed ({label})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PERSIST
# ─────────────────────────────────────────────────────────────────────────────

def _close_memmap_handle(st) -> None:
    """Close a memmap-backed datacube's file handle before the ref is dropped,
    else the OS keeps the mapping resident. Shared by ``free_memory`` and the
    cheaper ``release_scans``."""
    dc = getattr(st, "datacube", None)
    try:
        data = getattr(dc, "data", None)
        base = getattr(data, "base", None)
        for obj in (data, base):
            mm = getattr(obj, "_mmap", None) or (obj if "mmap" in type(obj).__name__.lower() else None)
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
    except Exception:
        pass


def release_scans(scans: list, *, log: Log = None) -> int:
    """Cheap, synchronous variant of :func:`free_memory` for the scan-switch path.

    Closes memmap handles and nulls the same heavy attributes as ``free_memory``
    (datacube / visualcube / vacuumcube / BVM histograms / probe), but skips the
    3x ``gc.collect()``, OS working-set trim, and CuPy pool free — those cost tens
    to hundreds of ms on a large heap, and the scan-switch handler must stay cheap
    (``qt_main.py`` ``_on_file_selected``: "Selection must stay CHEAP"). Figures and
    braggpeaks are intentionally left alone — those are governed by ``FigurePolicy``
    and the explicit "Free RAM" button respectively.

    Call :func:`free_memory` (already wired to "Free RAM" and to batch completion)
    to reclaim the actual OS/GPU memory once several scans have accumulated
    None-ed buffers.
    """
    n = 0
    for sc in (scans or []):
        st = getattr(sc, "state", None)
        if st is None:
            continue
        _close_memmap_handle(st)
        for a in ("datacube", "visualcube", "vacuumcube", "bvm_raw", "bvm_centered",
                  "dp_mean", "dp_max", "strainmap_full", "selected_disks", "probe"):
            if getattr(st, a, None) is not None:
                try:
                    setattr(st, a, None); n += 1
                except Exception:
                    pass
    if n:
        _log(log, f"Released {n} heavy buffer(s) from {len(scans)} inactive scan(s) (cheap pass).")
    return n


def free_memory(scans: list, *, drop_braggpeaks: bool = False, log: Log = None) -> dict:
    """Release the heavy in-memory buffers after a compute (the .mib datacube can be
    tens of GB). Drops state.datacube / visualcube / vacuumcube / BVM histograms (they
    re-load on demand), runs gc.collect(), and frees the CuPy GPU memory pools.

    The LIGHT ADF preview (scan.adf_cache) + the calibration on braggpeaks survive, so
    pickers/figures still work. ``drop_braggpeaks=True`` also releases the detected
    disks (free more RAM; they re-load from the braggpeaks.h5 when next needed).
    Returns {'buffers': n, 'gpu_freed_bytes': int}."""
    import gc
    n = 0
    for sc in (scans or []):
        st = getattr(sc, "state", None)
        if st is None:
            continue
        _close_memmap_handle(st)
        attrs = ["datacube", "visualcube", "vacuumcube", "bvm_raw", "bvm_centered",
                 "dp_mean", "dp_max", "strainmap_full", "selected_disks", "probe"]
        if drop_braggpeaks:
            attrs.append("braggpeaks")
        for a in attrs:
            if getattr(st, a, None) is not None:
                try:
                    setattr(st, a, None); n += 1
                except Exception:
                    pass
        try:                                   # py4DSTEM keeps products in a tree too
            if hasattr(st, "reset_data_products"):
                st.reset_data_products()
        except Exception:
            pass
    # several passes — CPython frees reference cycles incrementally, and dropping the
    # datacube can release big object graphs whose members only collect on a 2nd pass.
    for _ in range(3):
        gc.collect()
    try:
        import ctypes
        if sys.platform == "win32":            # Windows: trim the process working set
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        else:                                  # Linux glibc: return freed arena to the OS
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    gpu_freed = 0
    try:
        import cupy
        mp = cupy.get_default_memory_pool()
        before = mp.used_bytes()
        mp.free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        gpu_freed = int(max(0, before - mp.used_bytes()))
    except Exception:
        pass
    _log(log, f"Free RAM: released {n} heavy buffer(s) + gc×3 + OS trim"
              + (f" + CUDA pool ({gpu_freed/1e6:.0f} MB)" if gpu_freed else " + CUDA pool")
              + ". (Some residual RAM is normal — Python doesn't always return freed "
              "heap to the OS; the working-set trim helps on Windows.)")
    return {"buffers": n, "gpu_freed_bytes": gpu_freed}


def reset_bragg_calibration(scan: Scan, *, log: Log = None) -> None:
    """Reset ALL calibrations on this file's braggpeaks back to the uncalibrated state
    (origin / ellipse / Q-pixel / basis cleared). Reloads the braggpeaks fresh from
    its .h5 when available (cleanest); otherwise resets the calibration in place. Also
    clears the calibration checkpoints + the per-step calibration figures."""
    st = scan.ensure_state()
    reloaded = False
    if scan.braggpeaks_path:
        try:
            load_braggpeaks(scan, log=log)        # fresh, uncalibrated object
            reloaded = True
        except Exception as exc:
            _log(log, f"[{scan.name}] braggpeaks reload failed ({exc}); clearing in place")
    if not reloaded:
        bp = getattr(st, "braggpeaks", None)
        cal = getattr(bp, "calibration", None) if bp is not None else None
        if cal is not None and isinstance(getattr(cal, "_params", None), dict):
            for k in ("origin", "qx0", "qy0", "qx0_mean", "qy0_mean", "p_ellipse",
                      "a", "b", "theta", "QR_rotation_degrees", "QR_flip"):
                cal._params.pop(k, None)
            try:
                bp.setcal()
            except Exception:
                pass
    scan.cal_checkpoints = {}
    for k in ("origin", "ellipse", "q_pixel", "basis"):
        scan.figures.pop(k, None)
        clear_figure_spill(scan, keys=[k], delete_files=True)
    for a in ("strainmap_full", "bvm_raw", "bvm_centered"):
        if getattr(st, a, None) is not None:
            setattr(st, a, None)
    p = scan.params
    p.center_guess = [128.0, 128.0]
    p.origin_sampling = 2
    p.ellipse_enabled = False
    p.ellipse_q_range = [40, 60]
    p.ellipse_sampling = 1
    p.q_px_fitted = None
    p.qr_rotation = 0.0
    p.qr_flip = False
    p.basis_manual_enabled = False
    p.index_origin = 0
    p.index_g1 = 3
    p.index_g2 = 4
    scan.status = "pending"
    _log(log, f"[{scan.name}] bragg calibration reset to uncalibrated"
              + (" (reloaded braggpeaks)" if reloaded else " (cleared in place)"))


def load_datacube_and_probe(scan: Scan, *, log: Log = None) -> None:
    """One-shot for the detection tuner: load the raw 4D datacube AND compute the
    probe (from the vacuum file / configured source) so the 6-point detection can run
    without going back to Step 1."""
    load_datacube(scan, log=log)
    try:
        compute_probe(scan, log=log)
    except Exception as exc:
        _log(log, f"[{scan.name}] probe compute skipped: {exc} "
                  f"(set a vacuum file or probe source first).")
    _log(log, f"[{scan.name}] datacube + probe ready for detection.")


def save_results(scan: Scan, *, save_figures: bool = True, output_root: str | None = None,
                 vimg_cmap: str = "gray", log: Log = None) -> dict:
    """Persist strain tensors + ADF + manifest + figures (cell 45)."""
    st = scan.ensure_state()
    res = _artifacts().save_fast_artifacts(
        st, save_figures=save_figures, output_root=output_root, log=log)
    scan.results_dir = str(res.get("scan_dir", ""))
    scan.figures_dir = str(res.get("figures_dir", ""))
    # also save the raw virtual images (ADF/BF/DP mean/DP max) as .npy + plain PNGs
    if scan.results_dir:
        try:
            save_virtual_images(scan, Path(scan.results_dir) / "virtual_images",
                                cmap=vimg_cmap, log=log)
        except Exception as exc:
            _log(log, f"[{scan.name}] virtual-images save skipped: {exc}")
    # embed the full analysis metadata INTO the scan's .h5 (self-describing on reopen)
    try:
        embed_metadata_h5(scan, log=log)
    except Exception as exc:
        _log(log, f"[{scan.name}] metadata embed skipped: {exc}")
    return res


def _load_saved_figures(scan: "Scan", *, log: Log = None) -> int:
    """Load saved figure PNGs from the workspace ``figures/`` dir into scan.figures.

    A hydrated workspace has the strain/stress/calibration maps only as PNGs on disk
    (e.g. strain_without_roi.png, strain_with_roi.png, origin.png, q_pixel.png…), not
    as live matplotlib Figures — so the Report / per-column views showed nothing for
    them. We wrap each known PNG (canonical FIGURE_ORDER keys, so the duplicated
    strain_strain_*_0.png are ignored) in a Figure(imshow). Returns the count loaded.
    """
    fdir = Path(scan.figures_dir) if scan.figures_dir else None
    if not (fdir and fdir.is_dir()) and scan.results_dir:
        cand = Path(scan.results_dir) / "figures"
        fdir = cand if cand.is_dir() else None
    if not (fdir and fdir.is_dir()):
        return 0
    import matplotlib.image as mpimg
    from matplotlib.figure import Figure
    n = 0
    for key in FIGURE_ORDER:
        png = fdir / f"{key}.png"
        if not png.is_file():
            continue
        try:
            img = mpimg.imread(str(png))
            fig = Figure(figsize=(5, 4))
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            scan.figures[key] = fig
            n += 1
        except Exception as exc:
            _log(log, f"[{scan.name}] could not load figure {png.name}: {exc}")
    if n:
        _log(log, f"[{scan.name}] loaded {n} saved figure(s) from {fdir.name}/")
    return n


def load_results(scan: Scan, data_dir: str, *, log: Log = None) -> dict:
    """Hydrate a scan from a saved workspace — NO recompute. Also loads the saved
    figure PNGs (strain/stress maps etc.) so the Report shows them."""
    st = scan.ensure_state()
    res = _artifacts().load_fast_artifacts(st, data_dir, log=log)
    dd = Path(data_dir)
    scan.results_dir = str(dd.parent if dd.name == "data" else dd)
    fig_dir = Path(scan.results_dir) / "figures"
    scan.figures_dir = str(fig_dir) if fig_dir.is_dir() else ""
    scan.status = "done"
    _load_saved_figures(scan, log=log)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS — light, reads persisted strain arrays
# ─────────────────────────────────────────────────────────────────────────────

def compute_stress(scan: Scan, *, label: str = "without_roi",
                   mode: str = "plane_stress",
                   log: Log = None) -> dict | None:
    """Hooke's-law stress over a saved strain map. Light (no get_strain).

    Constants come from the scan's stress material + symmetry (cubic → 3
    independent; isotropic → C44 derived). Override per-scan by editing
    ``scan.params.stress_material`` / ``custom_stress`` / ``stress_symmetry``.
    """
    st = scan.ensure_state()
    c11_gpa, c12_gpa, c44_gpa = scan.params.stress_constants_gpa()
    c11, c12, c44 = c11_gpa * 1e9, c12_gpa * 1e9, c44_gpa * 1e9
    units = getattr(scan.params, "stress_units", "GPa") or "GPa"
    vmax = float(getattr(scan.params, "stress_vmax", 0.0) or 0.0)   # in display units; 0 = auto
    vmin_arg = -vmax if vmax > 0 else None
    vmax_arg = vmax if vmax > 0 else None
    try:
        result = _pipeline().compute_stress_analysis_step(
            st, label=label, mode=mode, c11_pa=c11, c12_pa=c12, c44_pa=c44,
            units=units, vmin_gpa=vmin_arg, vmax_gpa=vmax_arg, log=log)
        # register the stress-maps figure so the Report can show it (stress_<label>)
        fig = result.get("figure") if isinstance(result, dict) else None
        if fig is not None:
            register_figure(scan, f"stress_{label}", fig)
        return result
    except Exception as exc:
        _log(log, f"[{scan.name}] stress {label} failed: {exc}")
        return None


def extract_line_profiles(scan: Scan, lines: dict, *, line_width: int = 1) -> dict:
    """Resample strain/ADF along lines from saved arrays (cell 43). Light.

    lines: {line_id: [[x0,y0],[x1,y1]], ...}
    Returns {label: {line_id: {eyy,exx,exy,dist_px}}, "adf": {...}}.
    """
    st = scan.ensure_state()
    strain_raw = getattr(st, "strain_raw", {}) or {}
    adf = _artifacts()._adf_from_state(st)
    return _batch()._extract_line_profiles(strain_raw, adf, lines, line_width)


# ─────────────────────────────────────────────────────────────────────────────
# LINE TOOL — same region across N files: pick lines once on a template, propagate
# to every file, then re-propagate WITH per-file drift (dx,dy) so the lines land on
# the same PHYSICAL region. Lines are horizontal, full real-space width (x: 0→W-1);
# only the row y matters, and the drift shifts y per file (x already spans all).
# ─────────────────────────────────────────────────────────────────────────────

def _scan_width(scan: "Scan") -> int:
    """Real-space width W (columns) for full-width horizontal lines (x: 0→W-1)."""
    a = cached_adf(scan)
    if a is not None and getattr(a, "ndim", 0) == 2:
        return int(a.shape[1])
    rs = _real_space_shape(scan.state) if scan.state is not None else None
    return int(rs[1]) if rs else 512


def _scan_shape(scan: "Scan") -> tuple:
    """Real-space (H, W) for placing lines (full-width / full-height)."""
    a = cached_adf(scan)
    if a is not None and getattr(a, "ndim", 0) == 2:
        return (int(a.shape[0]), int(a.shape[1]))
    rs = _real_space_shape(scan.state) if scan.state is not None else None
    return (int(rs[0]), int(rs[1])) if rs else (512, 512)


def _scan_width(scan: "Scan") -> int:
    return _scan_shape(scan)[1]


def _spec_to_segment(spec: dict, H: int, W: int, dx: float = 0.0, dy: float = 0.0) -> list:
    """A typed line spec → a concrete segment [[x0,y0],[x1,y1]] for a (H,W) scan.

    Types: 'h' (horizontal, full width, key 'y'), 'v' (vertical, full height, key
    'x'), 'seg' (arbitrary, keys 'p0','p1'). Drift shifts BOTH x (by dx) and y (by
    dy) and CLAMPS to the image: x∈[0,W-1], y∈[0,H-1]. So a full-width row [0,W-1]
    with dx=+5 becomes [5, W-1] (the right end is clamped, not extended)."""
    W, H = int(W), int(H)

    def cx(x):
        return int(max(0, min(round(float(x)), W - 1)))

    def cy(y):
        return int(max(0, min(round(float(y)), H - 1)))

    t = spec.get("type", "h")
    if t == "h":
        y = cy(float(spec["y"]) + dy)
        return [[cx(0 + dx), y], [cx((W - 1) + dx), y]]
    if t == "v":
        x = cx(float(spec["x"]) + dx)
        return [[x, cy(0 + dy)], [x, cy((H - 1) + dy)]]
    p0, p1 = spec["p0"], spec["p1"]
    return [[cx(p0[0] + dx), cy(p0[1] + dy)], [cx(p1[0] + dx), cy(p1[1] + dy)]]


def _line_drift_shift(scan: "Scan", *, use_drift: bool) -> tuple:
    """(dx, dy) to place template lines on *scan* when drift CSV is loaded.

    Sign is **negated** vs the registration table: the CSV stores how the scan
    moved relative to the template; line coords must be shifted the other way to
    land on the same physical feature."""
    if not use_drift or not getattr(scan, "drift", None):
        return 0.0, 0.0
    return -float(scan.drift[0]), -float(scan.drift[1])


def propagate_template_lines(scans: list, line_specs: list, *, use_drift: bool = False,
                             log: Log = None) -> None:
    """Place the template line specs (h / v / seg) onto every scan as concrete
    segments. With ``use_drift`` each scan's segment is shifted by its own
    ``drift`` (dy, and dx for vertical/arbitrary) → same physical region per file."""
    specs = list(line_specs or [])
    for sc in scans:
        H, W = _scan_shape(sc)
        dx, dy = _line_drift_shift(sc, use_drift=use_drift)
        sc.lines = {f"L{i + 1}": _spec_to_segment(s, H, W, dx, dy)
                    for i, s in enumerate(specs)}
        _log(log, f"[{sc.name}] {len(sc.lines)} line(s) placed "
                  f"(drift={'ON' if use_drift else 'off'}, dx={dx:+.1f}, dy={dy:+.1f})")


def collect_line_ids(scans: list) -> set:
    ids: set = set()
    for sc in scans:
        ids.update((getattr(sc, "lines", None) or {}))
    return ids


def allocate_line_ids(scans: list, n: int) -> list:
    """Return *n* unused ids ``L1``, ``L2``, … not present on any scan."""
    taken = collect_line_ids(scans)
    out: list = []
    k = 1
    while len(out) < max(0, int(n)):
        lid = f"L{k}"
        if lid not in taken:
            out.append(lid)
            taken.add(lid)
        k += 1
    return out


def place_line_from_spec(scans: list, line_id: str, spec: dict, *,
                         template_scan: "Scan", use_drift: bool = False,
                         log: Log = None) -> None:
    """Merge one line onto each scan (does not wipe existing ``scan.lines``)."""
    lid = str(line_id)
    for sc in scans:
        H, W = _scan_shape(sc)
        if sc is template_scan:
            dx = dy = 0.0
        else:
            dx, dy = _line_drift_shift(sc, use_drift=use_drift)
        seg = _spec_to_segment(spec, H, W, dx, dy)
        lines = dict(getattr(sc, "lines", None) or {})
        lines[lid] = seg
        sc.lines = lines
        _log(log, f"[{sc.name}] line {lid} placed "
                  f"(drift={'ON' if use_drift and sc is not template_scan else 'off'})")


def register_live_line_report_figures(scans: list, line_ids: list, *,
                                      channel: str = "eyy", label: str = "without_roi",
                                      width: int = 3, log: Log = None) -> list[str]:
    """Materialize Live-line figures only under ``report_*`` keys (Send to Report).

    Does **not** register generic ``line_profiles`` / ``maps_with_lines`` — those
    stay on-demand. Returns the figure keys written on the first scan (for UI jump).
    """
    keys_written: list[str] = []
    for sc in scans:
        if not getattr(sc, "lines", None):
            continue
        for lid in line_ids:
            key = f"report_line_{lid}_{channel}_{label}"
            fig = build_single_line_map_figure(sc, lid, channel, label, width=width)
            if register_figure(sc, key, fig, force=True):
                if sc is scans[0]:
                    keys_written.append(key)
        prof_key = f"report_line_profiles_{channel}_{label}"
        fig = build_line_profiles_figure(sc, label, channel, width=width, register=False)
        if register_figure(sc, prof_key, fig, force=True):
            if sc is scans[0] and prof_key not in keys_written:
                keys_written.append(prof_key)
    for lid in line_ids:
        if not scans:
            break
        gkey = f"report_line_group_{lid}_{channel}_{label}"
        fig = build_grouped_line_figure(scans, lid, channel, label, width=width)
        if register_figure(scans[0], gkey, fig, force=True):
            keys_written.append(gkey)
    _log(log, f"Report (Send): {len(line_ids)} line(s) on {len(scans)} file(s), "
              f"width={width}px → {', '.join(keys_written) or '(none)'}")
    return keys_written


def scan_line_segments(scan: "Scan") -> list:
    """This scan's placed line segments [[x0,y0],[x1,y1]] (for the ADF overlay)."""
    return list((getattr(scan, "lines", None) or {}).values())


def load_drift_csv(path: str, scans: list, *, log: Log = None) -> dict:
    """Load a per-file drift table (the plugin's CSV) and assign ``scan.drift=(dx,dy)``.

    Tolerant to column names: name ∈ {file,filename,name,scan,stem,sample}, dx ∈
    {dx,shift_x,drift_x,x,...}, dy ∈ {dy,shift_y,drift_y,y,...}. Matches scans by
    name / file stem; if there is no recognizable name column, assigns by row order.
    Returns the parsed {key:(dx,dy)} map. (Refine once the real CSV is shared.)
    """
    import pandas as pd
    df = pd.read_csv(path)
    cols = {str(c).lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    def pick_sub(*subs, avoid=()):                # substring fallback over column names
        for c in cols:
            if any(s in c for s in subs) and not any(a in c for a in avoid):
                return cols[c]
        return None

    name_col = (pick("scan_id", "file", "filename", "name", "scan", "stem", "sample", "label")
                or pick_sub("scan", "file", "name"))
    dx_col = (pick("shift_dx_px", "dx_px", "shift_dx", "drift_dx", "dx",
                   "shift_x", "drift_x", "shiftx", "driftx", "dxpx", "x")
              or pick_sub("dx"))
    dy_col = (pick("shift_dy_px", "dy_px", "shift_dy", "drift_dy", "dy",
                   "shift_y", "drift_y", "shifty", "drifty", "dypx", "y")
              or pick_sub("dy"))
    _log(log, f"drift CSV columns → name={name_col!r} dx={dx_col!r} dy={dy_col!r}")
    parsed: dict = {}
    rows = list(df.iterrows())
    for _idx, row in rows:
        key = str(row[name_col]).strip() if name_col is not None else None
        dx = float(row[dx_col]) if dx_col is not None else 0.0
        dy = float(row[dy_col]) if dy_col is not None else 0.0
        parsed[key if key is not None else f"#{_idx}"] = (dx, dy)

    if name_col is not None:
        for sc in scans:
            cand = {c for c in (sc.name,
                                Path(sc.raw_path).stem if sc.raw_path else "",
                                Path(sc.h5_path).stem if sc.h5_path else "",
                                Path(sc.braggpeaks_path).stem if sc.braggpeaks_path else "")
                    if c}
            hit = None
            for k, val in parsed.items():
                ks = Path(str(k)).stem or str(k)
                if str(k) in cand or ks in cand or any(ks and (ks in c or c in ks) for c in cand):
                    hit = val
                    break
            if hit is not None:
                sc.drift = hit
                _log(log, f"[{sc.name}] drift: dx={hit[0]:+.2f} dy={hit[1]:+.2f}")
            else:
                _log(log, f"[{sc.name}] no drift row matched (lines will use dy=0).")
    else:                                   # no name column → assign by order
        vals = list(parsed.values())
        for sc, v in zip(scans, vals):
            sc.drift = v
            _log(log, f"[{sc.name}] drift (by order): dx={v[0]:+.2f} dy={v[1]:+.2f}")
    return parsed


def _extract_line_rows(cand) -> list:
    """Pull horizontal-line row y's from a variety of JSON line shapes."""
    rows: list[float] = []
    items = list(cand.values()) if isinstance(cand, dict) else list(cand or [])
    for v in items:
        y = None
        if isinstance(v, (int, float)):
            y = float(v)
        elif isinstance(v, dict):
            if "y" in v:
                y = float(v["y"])
            elif "points" in v and v["points"]:
                p0 = v["points"][0]
                y = float(p0[1]) if isinstance(p0, (list, tuple)) and len(p0) >= 2 else None
        elif isinstance(v, (list, tuple)) and v:
            first = v[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:   # [[x0,y0],[x1,y1]]
                y = float(first[1])
            elif all(isinstance(t, (int, float)) for t in v[:2]) and len(v) >= 2:  # [x,y]/[x0,y0,x1,y1]
                y = float(v[1])
            elif isinstance(first, (int, float)):
                y = float(first)
        if y is not None:
            rows.append(int(round(y)))
    # de-dup, keep order
    seen, out = set(), []
    for y in rows:
        if y not in seen:
            seen.add(y); out.append(y)
    return out


def _normalize_segments(seg_dict) -> dict:
    """{line_id: [[x0,y0],[x1,y1]]} from per-line shapes: [[x0,y0],[x1,y1]] or
    {x0,y0,x1,y1} or {points:[[x,y],…]}."""
    out: dict = {}
    for lid, v in (seg_dict or {}).items():
        p = None
        try:
            if isinstance(v, dict):
                if {"x0", "y0", "x1", "y1"} <= set(v):
                    p = [[float(v["x0"]), float(v["y0"])], [float(v["x1"]), float(v["y1"])]]
                elif "points" in v and len(v["points"]) >= 2:
                    a, b = v["points"][0], v["points"][1]
                    p = [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]]
            elif isinstance(v, (list, tuple)) and len(v) == 2 \
                    and all(isinstance(t, (list, tuple)) for t in v):
                p = [[float(v[0][0]), float(v[0][1])], [float(v[1][0]), float(v[1][1])]]
        except Exception:
            p = None
        if p is not None:
            out[str(lid)] = p
    return out


def _match_scan_key(per: dict, scan: "Scan"):
    """Find the per-scan key in ``per`` that matches a scan (by name / file stem)."""
    cand = {c for c in (scan.name,
                        Path(scan.raw_path).stem if scan.raw_path else "",
                        Path(scan.h5_path).stem if scan.h5_path else "",
                        Path(scan.braggpeaks_path).stem if scan.braggpeaks_path else "")
            if c}
    for k in per:                                  # exact name / stem
        ks = Path(str(k)).stem or str(k)
        if str(k) in cand or ks in cand:
            return k
    for k in per:                                  # substring fallback
        for c in cand:
            if c and (c in str(k) or str(k) in c):
                return k
    return None


def load_lines_json(path: str, scans: list, *, log: Log = None) -> dict:
    """Load line definitions from a JSON (the same file used by the data loader).

    Handles the real format: ``line_profiles_per_scan`` ({scan_name: {L: [[x0,y0],
    [x1,y1]]}}) → each scan gets its OWN lines (already drift-adjusted per file),
    assigned by name/stem; and ``fixed_line_profiles`` / lines / line_positions /
    line_profiles_px / line_profiles (template, same rows for all). Per-scan segments
    are written straight to ``scan.lines``. Returns
    ``{"mode": per_scan|template, "rows": [y…], "assigned": n}``.
    """
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    _LINE_KEYS = ("lines", "line_positions", "line_profiles_px",
                  "fixed_line_profiles", "fixed_line_profiles_px", "line_profiles")

    def _find(d):
        if not isinstance(d, dict):
            return None
        for k in _LINE_KEYS:
            if k in d and d[k]:
                return d[k]
        for nest in ("profile", "profiles", "line_settings"):
            sub = d.get(nest)
            if isinstance(sub, dict):
                hit = _find(sub)
                if hit:
                    return hit
        return None

    # 1) per-scan lines (each file its own, drift-adjusted) — written to scan.lines
    per = data.get("line_profiles_per_scan") if isinstance(data, dict) else None
    if isinstance(per, dict) and per:
        assigned = 0
        for sc in scans:
            k = _match_scan_key(per, sc)
            if k is not None:
                segs = _normalize_segments(per[k])
                if segs:
                    sc.lines = segs
                    assigned += 1
                    _log(log, f"[{sc.name}] {len(segs)} line(s) from JSON (per-scan '{k}')")
        rows = _extract_line_rows(data.get("fixed_line_profiles") or _find(data))
        _log(log, f"Lines JSON: per-scan, assigned to {assigned}/{len(scans)} file(s).")
        return {"mode": "per_scan", "rows": rows, "assigned": assigned}

    # 2) template lines (same rows for all → caller propagates)
    cand = _find(data)
    rows = _extract_line_rows(cand) if cand is not None else []
    _log(log, f"Lines JSON: template rows {rows if rows else 'none found'}.")
    return {"mode": "template", "rows": rows, "assigned": 0}


# ─────────────────────────────────────────────────────────────────────────────
# AREA ROI — same region across N files + per-file drift (parallel to lines)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_roi(v) -> list:
    """A variety of ROI shapes → [x0, x1, y0, y1] (sorted). Accepts
    [x0,x1,y0,y1], {x0,x1,y0,y1}, [[x0,y0],[x1,y1]] (corners) or {points:[…]}."""
    try:
        if isinstance(v, dict):
            if {"x0", "x1", "y0", "y1"} <= set(v):
                xs = [float(v["x0"]), float(v["x1"])]; ys = [float(v["y0"]), float(v["y1"])]
            elif "points" in v and len(v["points"]) >= 2:
                a, b = v["points"][0], v["points"][1]
                xs = [float(a[0]), float(b[0])]; ys = [float(a[1]), float(b[1])]
            else:
                return []
        elif isinstance(v, (list, tuple)) and len(v) == 4 \
                and all(isinstance(t, (int, float)) for t in v):
            return [float(min(v[0], v[1])), float(max(v[0], v[1])),
                    float(min(v[2], v[3])), float(max(v[2], v[3]))]
        elif isinstance(v, (list, tuple)) and len(v) == 2 \
                and all(isinstance(t, (list, tuple)) for t in v):       # corners
            xs = [float(v[0][0]), float(v[1][0])]; ys = [float(v[0][1]), float(v[1][1])]
        else:
            return []
    except Exception:
        return []
    return [min(xs), max(xs), min(ys), max(ys)]


def _roi_shift_clamp(bounds: list, H: int, W: int, dx: float = 0.0, dy: float = 0.0) -> list:
    """Shift an ROI [x0,x1,y0,y1] by (dx,dy) and CLAMP to the image: x∈[0,W-1],
    y∈[0,H-1] (same condition as the lines — never exceed the image limit)."""
    x0, x1, y0, y1 = (float(b) for b in bounds)

    def cx(x):
        return int(max(0, min(round(x), int(W) - 1)))

    def cy(y):
        return int(max(0, min(round(y), int(H) - 1)))

    nx0, nx1 = cx(x0 + dx), cx(x1 + dx)
    ny0, ny1 = cy(y0 + dy), cy(y1 + dy)
    return [min(nx0, nx1), max(nx0, nx1), min(ny0, ny1), max(ny0, ny1)]


def propagate_template_roi(scans: list, roi_bounds: list, *, use_drift: bool = False,
                           log: Log = None) -> None:
    """Place the template area-ROI onto every scan, shifted by each scan's own
    drift (when ``use_drift``) and clamped to the image — same physical region per
    file. Mirrors ``propagate_template_lines`` for a single rectangle."""
    bounds = _normalize_roi(roi_bounds)
    if not bounds:
        _log(log, "Area ROI: nothing to propagate (no template ROI).")
        return
    for sc in scans:
        H, W = _scan_shape(sc)
        dx, dy = _line_drift_shift(sc, use_drift=use_drift)
        sc.area_roi = _roi_shift_clamp(bounds, H, W, dx, dy)
        _log(log, f"[{sc.name}] area ROI {sc.area_roi} "
                  f"(drift={'ON' if use_drift else 'off'}, dx={dx:+.1f}, dy={dy:+.1f})")


def scan_area_roi(scan: "Scan") -> list:
    """This scan's placed area ROI [x0,x1,y0,y1] (or [] if none).

    Legacy single-ROI accessor. Returns the first multi-ROI when ``area_rois`` is
    populated so old callers still see *an* ROI; prefer ``scan_area_rois`` for the
    full set."""
    multi = scan_area_rois(scan)
    if multi:
        return list(next(iter(sorted(multi.items())))[1])
    return list(getattr(scan, "area_roi", None) or [])


def scan_area_rois(scan: "Scan") -> dict:
    """The scan's multi analysis ROIs {roi_id: [x0,x1,y0,y1]} (source of truth).

    Migrates a legacy single ``area_roi`` to ``{"R1": bounds}`` on read, so projects
    saved before multi-ROI keep working without a separate migration pass."""
    if scan is None:
        return {}
    d = getattr(scan, "area_rois", None) or {}
    out: dict = {}
    for k, v in d.items():
        b = _normalize_roi(v)
        if b:
            out[str(k)] = b
    if out:
        return out
    leg = _normalize_roi(getattr(scan, "area_roi", None))
    return {"R1": leg} if leg else {}


def collect_roi_ids(scans: list) -> set:
    ids: set = set()
    for sc in scans:
        ids.update(scan_area_rois(sc))
    return ids


def allocate_roi_ids(scans: list, n: int) -> list:
    """Return *n* unused ids ``R1``, ``R2``, … not present on any scan."""
    taken = collect_roi_ids(scans)
    out: list = []
    k = 1
    while len(out) < max(0, int(n)):
        rid = f"R{k}"
        if rid not in taken:
            out.append(rid)
            taken.add(rid)
        k += 1
    return out


def place_roi_from_spec(scans: list, roi_id: str, bounds: list, *,
                        template_scan: "Scan", use_drift: bool = False,
                        log: Log = None) -> None:
    """Merge one area ROI onto each scan (does not wipe existing ``scan.area_rois``).
    Mirrors ``place_line_from_spec``: the template scan keeps the picked rectangle,
    every other scan gets it shifted by its own drift and clamped to the image."""
    rid = str(roi_id)
    tpl = _normalize_roi(bounds)
    if not tpl:
        return
    for sc in scans:
        H, W = _scan_shape(sc)
        if sc is template_scan:
            dx = dy = 0.0
        else:
            dx, dy = _line_drift_shift(sc, use_drift=use_drift)
        rois = dict(scan_area_rois(sc))
        rois[rid] = _roi_shift_clamp(tpl, H, W, dx, dy)
        sc.area_rois = rois
        _log(log, f"[{sc.name}] ROI {rid} placed "
                  f"(drift={'ON' if use_drift and sc is not template_scan else 'off'})")


def propagate_template_rois(scans: list, rois: dict, *, use_drift: bool = False,
                            log: Log = None) -> None:
    """Place a whole template ROI set {roi_id: bounds} onto every scan, each ROI
    shifted by the scan's own drift (when ``use_drift``) and clamped to the image —
    same physical regions per file. Mirrors ``propagate_template_lines``."""
    specs = {str(k): _normalize_roi(v) for k, v in (rois or {}).items()}
    specs = {k: v for k, v in specs.items() if v}
    if not specs:
        _log(log, "Area ROIs: nothing to propagate (no template ROIs).")
        return
    for sc in scans:
        H, W = _scan_shape(sc)
        dx, dy = _line_drift_shift(sc, use_drift=use_drift)
        sc.area_rois = {rid: _roi_shift_clamp(b, H, W, dx, dy) for rid, b in specs.items()}
        _log(log, f"[{sc.name}] {len(specs)} ROI(s) placed "
                  f"(drift={'ON' if use_drift else 'off'}, dx={dx:+.1f}, dy={dy:+.1f})")


def scan_area_roi_segments(scan: "Scan") -> dict:
    """Each ROI as 4 closed segments {roi_id: [[ [x0,y0],[x1,y0] ], …]} for overlays."""
    out: dict = {}
    for rid, b in scan_area_rois(scan).items():
        x0, x1, y0, y1 = b
        out[rid] = [[[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]],
                    [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]]]
    return out


def scan_display_roi(scan: "Scan") -> list:
    """ROI to draw on ADF previews.

    Prefer the analysis/area ROI used by line profiles when present; otherwise
    show the calibration ROI from the parameter table (``params.roi_bounds``).
    """
    if scan is None:
        return []
    roi = scan_area_roi(scan)
    if len(roi) == 4:
        return roi
    try:
        b = list(getattr(scan.params, "roi_bounds", None) or [])
    except Exception:
        b = []
    return b if len(b) == 4 else []


def roi_segments(scan: "Scan") -> list:
    """The area ROI as 4 closed segments [[x0,y0],[x1,y1]] (for thumbnail overlays)."""
    b = scan_display_roi(scan)
    if len(b) != 4:
        return []
    x0, x1, y0, y1 = b
    return [[[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]],
            [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]]]


def load_roi_json(path: str, scans: list, *, log: Log = None) -> dict:
    """Load the analysis ROI from the SAME JSON the loader/line tool uses.

    Tolerant: ``roi_per_scan`` / ``area_roi_per_scan`` ({scan_name: ROI}) → each
    scan its OWN ROI (assigned by name/stem, already drift-adjusted per file); else
    ``fixed_roi`` / ``area_roi`` / ``roi`` / ``roi_bounds`` / ``profile_roi`` as a
    template (same region for all → caller propagates). Returns
    ``{"mode": per_scan|template, "bounds": [x0,x1,y0,y1]|[], "assigned": n}``."""
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    _ROI_PER = ("roi_per_scan", "area_roi_per_scan", "rois_per_scan", "fixed_roi_per_scan")
    _ROI_KEYS = ("area_roi", "fixed_roi", "roi", "roi_bounds", "profile_roi", "analysis_roi")

    def _find_template(d):
        if not isinstance(d, dict):
            return None
        for k in _ROI_KEYS:
            if k in d and d[k]:
                b = _normalize_roi(d[k])
                if b:
                    return b
        for nest in ("profile", "profiles", "roi_settings", "area_settings", "line_settings"):
            sub = d.get(nest)
            if isinstance(sub, dict):
                hit = _find_template(sub)
                if hit:
                    return hit
        return None

    per = None
    if isinstance(data, dict):
        for k in _ROI_PER:
            if isinstance(data.get(k), dict) and data[k]:
                per = data[k]
                break
    if isinstance(per, dict) and per:                  # per-scan ROIs
        assigned = 0
        for sc in scans:
            k = _match_scan_key(per, sc)
            if k is not None:
                b = _normalize_roi(per[k])
                if b:
                    sc.area_roi = b
                    assigned += 1
                    _log(log, f"[{sc.name}] area ROI from JSON (per-scan '{k}'): {b}")
        tpl = _find_template(data) or []
        _log(log, f"ROI JSON: per-scan, assigned to {assigned}/{len(scans)} file(s).")
        return {"mode": "per_scan", "bounds": tpl, "assigned": assigned}

    tpl = _find_template(data) or []
    _log(log, f"ROI JSON: template ROI {tpl if tpl else 'none found'}.")
    return {"mode": "template", "bounds": tpl, "assigned": 0}


def build_lines_overlay_figure(scan: "Scan"):
    """ADF with this scan's horizontal lines overlaid (preview / Report)."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle as plt_Rectangle
    a = cached_adf(scan)
    fig = Figure(figsize=(5.2, 4.2), constrained_layout=True)
    ax = fig.add_subplot(111)
    if a is None:
        ax.text(0.5, 0.5, "no ADF for this scan", ha="center", va="center"); ax.axis("off")
        return fig
    v = a[a > 0]
    vmin, vmax = ((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                 if v.size else (float(a.min()), float(a.max() or 1)))
    ax.imshow(a, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    for i, (lid, seg) in enumerate(sorted((getattr(scan, "lines", None) or {}).items())):
        (x0, y0), (x1, y1) = seg
        c = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
        ax.plot([x0, x1], [y0, y1], color=c, lw=1.4)
        ax.text(x0 + 2, y0 - 2, lid, color=c, fontsize=7, va="bottom")
    rois = scan_area_rois(scan)                        # ALL multi analysis ROIs
    if rois:
        for i, rid in enumerate(sorted(rois)):
            rx0, rx1, ry0, ry1 = rois[rid]
            c = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
            ax.add_patch(plt_Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                                       fill=False, edgecolor=c, lw=1.6))
            ax.text(rx0 + 2, ry1 - 2, rid, color=c, fontsize=7, va="top")
    else:                                              # fallback: calibration ROI (dashed cyan)
        roi = scan_display_roi(scan)
        if len(roi) == 4:
            rx0, rx1, ry0, ry1 = roi
            ax.add_patch(plt_Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                                       fill=False, edgecolor="#00E5FF", lw=1.6, ls="--"))
            ax.text(rx0 + 2, ry1 - 2, "ROI", color="#00E5FF", fontsize=7, va="top")
    dr = getattr(scan, "drift", None)
    ax.set_title(f"{scan.name} — {len(getattr(scan, 'lines', None) or {})} line(s)"
                 + (f" + {len(rois)} ROI(s)" if rois else "")
                 + (f"  (drift dx={dr[0]:+.1f}, dy={dr[1]:+.1f})" if dr else "  (no drift)"),
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    register_figure(scan, "lines", fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LINE PROFILES — resample strain/ADF along the placed lines (for the Report)
# ─────────────────────────────────────────────────────────────────────────────

_CH_IDX = {"eyy": 0, "exx": 1, "exy": 2}                 # strain channels (in strain_raw[...,idx])
_STRESS_KEY = {"sxx": "sigma_xx", "syy": "sigma_yy", "sxy": "sigma_xy"}  # stress (stress_tensors_pa, Pa)
_CH_LABEL = {"eyy": "ε_yy (%)", "exx": "ε_xx (%)", "exy": "ε_xy (%)", "adf": "ADF",
             "sxx": "σ_xx", "syy": "σ_yy", "sxy": "σ_xy",
             "orientation": "θ (°)", "theta": "θ (°)"}
_ORIENT_CH = frozenset({"orientation", "theta"})


def _stress_scale(scan: "Scan"):
    """(Pa→display divisor, unit string) from the scan's stress_units."""
    units = (getattr(scan.params, "stress_units", "GPa") or "GPa")
    return (1e6, "MPa") if str(units).upper() == "MPA" else (1e9, "GPa")


def _channel_label(scan: "Scan", channel: str) -> str:
    if channel in _STRESS_KEY:
        return f"{_CH_LABEL[channel]} ({_stress_scale(scan)[1]})"
    return _CH_LABEL.get(channel, channel)


def channel_clim(scan: "Scan", channel: str) -> tuple[float, float] | None:
    """Symmetric or explicit (vmin, vmax) from GUI params — shared across panels.

    Strain ε → ``params.vrange`` (%). Orientation → ``params.vrange_theta`` (°).
    Stress → ±``params.stress_vmax`` when > 0; otherwise None (caller may auto).
    """
    p = getattr(scan, "params", None)
    if p is None:
        return None
    ch = str(channel).lower()
    if ch in _CH_IDX:
        vr = list(getattr(p, "vrange", None) or [-5.0, 5.0])
        return float(vr[0]), float(vr[1])
    if ch in _ORIENT_CH:
        vr = list(getattr(p, "vrange_theta", None) or [-5.0, 5.0])
        return float(vr[0]), float(vr[1])
    if ch in _STRESS_KEY:
        vmax = float(getattr(p, "stress_vmax", 0.0) or 0.0)
        if vmax > 0:
            return -vmax, vmax
        return None
    return None


def build_channel_panel_figure(scan: "Scan", channel: str, label: str = "without_roi",
                               *, title: str | None = None):
    """One-panel map figure for Report tree channel leaves / export (GUI vranges)."""
    import matplotlib.pyplot as plt

    arr = channel_map_2d(scan, channel, label)
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    cmap = str(getattr(scan.params, "strain_cmap", None) or "RdBu_r")
    if str(channel).lower() in _ORIENT_CH:
        cmap = str(getattr(scan.params, "strain_cmap_theta", None) or "PRGn")
    clim = channel_clim(scan, channel)
    if clim is None:
        finite = a[np.isfinite(a)]
        if finite.size:
            vmax = float(np.nanpercentile(np.abs(finite), 98)) or 1.0
            clim = (-vmax, vmax)
        else:
            clim = (-1.0, 1.0)
    vmin, vmax = clim
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title or f"{scan.name} — {_channel_label(scan, channel)} ({label})")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def map_key_label_channel(map_key: str) -> tuple[str, str]:
    """``strain_with_roi`` → (``with_roi``, ``strain``); stress similarly."""
    label = "with_roi" if "with_roi" in map_key else "without_roi"
    kind = "stress" if map_key.startswith("stress") else "strain"
    return label, kind



def _line_samples(arr2d, seg, *, width: int = 3, n: int | None = None):
    """Sample a 2D array along a segment (bilinear, averaged over ``width`` parallel
    offsets). Returns (dist_px, values)."""
    from scipy.ndimage import map_coordinates
    (x0, y0), (x1, y1) = seg
    L = float(np.hypot(x1 - x0, y1 - y0))
    npts = int(n or max(2, round(L) + 1))
    t = np.linspace(0.0, 1.0, npts)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    if L > 0:
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        px, py = -uy, ux                      # perpendicular unit
    else:
        px = py = 0.0
    w = max(1, int(width))
    offs = np.arange(w) - (w - 1) / 2.0
    acc = np.zeros(npts, dtype=float)
    arr = np.asarray(arr2d, dtype=float)
    for o in offs:
        acc += map_coordinates(arr, [ys + o * py, xs + o * px], order=1, mode="nearest")
    return t * L, acc / len(offs)


def line_profiles(scan: "Scan", label: str = "without_roi", *, width: int = 3) -> dict:
    """{line_id: {dist, eyy, exx, exy, adf}} resampled along ``scan.lines`` from the
    saved strain map (``label``, channels in %) and the ADF."""
    from fast_artifacts import _as_hw3
    st = scan.ensure_state()
    hw3 = _as_hw3((getattr(st, "strain_raw", {}) or {}).get(label))
    stress = (getattr(st, "stress_tensors_pa", {}) or {}).get(label)
    sdiv, _u = _stress_scale(scan)
    adf = cached_adf(scan)
    out: dict = {}
    for lid, seg in (getattr(scan, "lines", None) or {}).items():
        d: dict = {}
        if hw3 is not None:
            for ch, idx in _CH_IDX.items():
                dist, v = _line_samples(hw3[..., idx], seg, width=width)
                d["dist"] = dist
                d[ch] = v * 100.0
        if stress:                                   # stress maps too (σ in GPa/MPa)
            for ch, key in _STRESS_KEY.items():
                arr = stress.get(key)
                if arr is not None:
                    dist, v = _line_samples(np.asarray(arr, dtype=float), seg, width=width)
                    d.setdefault("dist", dist)
                    d[ch] = v / sdiv
        if adf is not None:
            dist, v = _line_samples(adf, seg, width=width)
            d.setdefault("dist", dist)
            d["adf"] = v
        if d:
            out[lid] = d
    return out


def build_line_profiles_figure(scan: "Scan", label: str = "without_roi",
                               channel: str = "eyy", *, width: int = 3,
                               register: bool = True):
    """Top: the chosen map (strain channel or ADF) with the lines overlaid.
    Bottom: that channel's profile vs distance (px), one curve per line."""
    from matplotlib.figure import Figure
    from fast_artifacts import _as_hw3
    prof = line_profiles(scan, label, width=width)
    st = scan.ensure_state()
    hw3 = _as_hw3((getattr(st, "strain_raw", {}) or {}).get(label))
    stress = (getattr(st, "stress_tensors_pa", {}) or {}).get(label)
    sdiv, _u = _stress_scale(scan)
    adf = cached_adf(scan)
    if channel == "adf":
        base = adf
    elif channel in _CH_IDX:
        base = hw3[..., _CH_IDX[channel]] * 100.0 if hw3 is not None else None
    elif channel in _STRESS_KEY:
        base = (np.asarray(stress[_STRESS_KEY[channel]], dtype=float) / sdiv
                if (stress and _STRESS_KEY[channel] in stress) else None)
    else:
        base = None
    ylab = _channel_label(scan, channel)
    fig = Figure(figsize=(7.6, 7.2), constrained_layout=True)
    ax_map = fig.add_subplot(2, 1, 1)
    if base is not None:
        if channel == "adf":
            v = base[base > 0]
            lo, hi = ((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                      if v.size else (float(base.min()), float(base.max() or 1)))
            ax_map.imshow(base, cmap="gray", vmin=lo, vmax=hi, origin="upper")
        else:
            m = float(np.nanpercentile(np.abs(base), 95)) or 1.0
            ax_map.imshow(base, cmap="RdBu_r", vmin=-m, vmax=m, origin="upper")
    ax_map.set_title(f"{scan.name} — {ylab} ({label})", fontsize=9)
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_p = fig.add_subplot(2, 1, 2)
    for i, (lid, seg) in enumerate(sorted((getattr(scan, "lines", None) or {}).items())):
        c = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
        (x0, y0), (x1, y1) = seg
        ax_map.plot([x0, x1], [y0, y1], color=c, lw=1.3)
        ax_map.text(x0 + 2, y0 - 2, lid, color=c, fontsize=7)
        pr = prof.get(lid, {})
        if channel in pr and "dist" in pr:
            ax_p.plot(pr["dist"], pr[channel], color=c, lw=1.2, label=lid)
    ax_p.set_xlabel("distance (px)"); ax_p.set_ylabel(ylab)
    ax_p.grid(alpha=0.3)
    if prof:
        ax_p.legend(fontsize=7, ncol=4)
    ax_p.set_title("Line profiles", fontsize=9)
    if register:
        register_figure(scan, "line_profiles", fig)
    return fig


def grouped_line_profiles(scans: list, line_id: str, channel: str = "eyy",
                          label: str = "without_roi", *, width: int = 3) -> dict:
    """The SAME line across files. Returns {series:[(name,dist,vals)], common_dist,
    mean_curve, std_curve, per_file_mean, per_file_std, pooled, names,
    summary{mean,std,sem,cv_pct,ci_lo,ci_hi,n}} — summary stats are across the
    per-file line means; per_file_std is each file's intra-line (along-distance)
    std and pooled concatenates every file's curve values (run-level spread)."""
    series = []
    for sc in scans:
        pr = line_profiles(sc, label, width=width).get(line_id)
        if pr and channel in pr and "dist" in pr:
            series.append((sc.name, np.asarray(pr["dist"]), np.asarray(pr[channel])))
    res: dict = {"series": series, "names": [s[0] for s in series]}
    if not series:
        res.update({"common_dist": np.zeros(0), "mean_curve": np.zeros(0),
                    "std_curve": np.zeros(0), "per_file_mean": [], "per_file_std": [],
                    "pooled": np.zeros(0), "summary": {}})
        return res
    m = min(len(v) for _n, _d, v in series)
    common_dist = series[0][1][:m]
    stack = np.vstack([v[:m] for _n, _d, v in series])
    res["common_dist"] = common_dist
    res["mean_curve"] = stack.mean(axis=0)
    res["std_curve"] = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(m)
    per = [float(np.mean(v)) for _n, _d, v in series]
    res["per_file_mean"] = per
    res["per_file_std"] = [float(np.std(v[:m], ddof=1)) if m > 1 else 0.0
                           for _n, _d, v in series]
    res["pooled"] = np.concatenate([v[:m] for _n, _d, v in series])
    res["summary"] = _summary_stats(np.asarray(per))
    return res


def _summary_stats(arr) -> dict:
    """Mean / Std / SE of mean / Coefficient of Variation / 95% CI of the mean."""
    from scipy import stats as _st
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return {"n": 0}
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n else 0.0
    ci = float(_st.t.ppf(0.975, n - 1)) * sem if n > 1 else 0.0
    cv = (std / abs(mean) * 100.0) if abs(mean) > 1e-12 else float("nan")
    return {"mean": mean, "std": std, "sem": sem, "cv_pct": cv,
            "ci95_lo": mean - ci, "ci95_hi": mean + ci, "n": n}


def build_grouped_line_figure(scans: list, line_id: str, channel: str = "eyy",
                              label: str = "without_roi", *, width: int = 3):
    """Three panels for the SAME line across files:
    1) distance-domain overlay of each file's profile + mean±std band;
    2) per-file mean ± intra-file std (point/errorbar) plus the across-file
       mean±std band and the summary stats box — the "run-level" view;
    3) pooled (every file, every distance sample) value histogram — the
       run-level spread, mirrors ``build_grouped_roi_figure``'s histogram."""
    from matplotlib.figure import Figure
    g = grouped_line_profiles(scans, line_id, channel, label, width=width)
    fig = Figure(figsize=(6.8, 13.2), constrained_layout=True)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    series, cd = g["series"], g["common_dist"]
    if not series:
        for ax in (ax1, ax2, ax3):
            ax.text(0.5, 0.5, "No profiles — set lines + compute strain first.",
                    ha="center", va="center"); ax.axis("off")
        return fig
    mean_c, std_c = g["mean_curve"], g["std_curve"]
    m = len(cd)
    clab = _channel_label(scans[0], channel) if scans else _CH_LABEL.get(channel, channel)

    for i, (name, _d, v) in enumerate(series):
        c = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
        ax1.plot(cd, v[:m], color=c, lw=1.0, alpha=0.8, label=name[:14])
    ax1.plot(cd, mean_c, color="black", lw=2.0, label="mean")
    ax1.fill_between(cd, mean_c - std_c, mean_c + std_c, color="0.5", alpha=0.25, label="±std")
    ax1.set_xlabel("distance (px)"); ax1.set_ylabel(clab)
    ax1.set_title(f"{line_id} across {len(series)} file(s) — {clab} ({label})", fontsize=9)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=6, ncol=2)

    names = g["names"]
    per, per_std = g["per_file_mean"], g["per_file_std"]
    s = g["summary"]
    mean = s.get("mean", float(np.mean(per)))
    std = s.get("std", 0.0)
    xs = range(len(names))
    for i, (mv, sv) in enumerate(zip(per, per_std)):
        c = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
        ax2.errorbar([i], [mv], yerr=[sv], fmt="o", color=c, capsize=3, elinewidth=1)
    ax2.axhline(mean, color="black", lw=2.0, label="mean")
    ax2.axhspan(mean - std, mean + std, color="0.5", alpha=0.2, label="±std")
    ax2.set_xticks(list(xs)); ax2.set_xticklabels([n[:14] for n in names], rotation=30, ha="right")
    ax2.set_ylabel(clab)
    ax2.set_title("Per-file mean ± intra-file std", fontsize=9)
    ax2.grid(alpha=0.3, axis="y"); ax2.legend(fontsize=7)
    if s.get("n"):
        txt = (f"across-file mean of line means:\nMean={s['mean']:.4g}  Std={s['std']:.4g}\n"
               f"SE={s['sem']:.4g}  CV={s['cv_pct']:.3g}%\n"
               f"95% CI=[{s['ci95_lo']:.4g}, {s['ci95_hi']:.4g}]  n={s['n']}")
        ax2.text(0.02, 0.98, txt, transform=ax2.transAxes, va="top", ha="left", fontsize=7,
                 bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#999", alpha=0.7))

    pooled = g["pooled"]
    if pooled.size:
        lo, hi = (float(np.percentile(pooled, 0.5)), float(np.percentile(pooled, 99.5)))
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            lo, hi = float(pooled.min()), float(pooled.max() or (lo + 1.0))
        bins = np.linspace(lo, hi, 41)
        ax3.hist(pooled, bins=bins, histtype="step", density=True, color="#3cb44b", lw=1.6)
        pmean = float(np.mean(pooled))
        pstd = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0
        ax3.axvline(pmean, color="#3cb44b", lw=1.0, ls="--", alpha=0.8)
        pcv = (pstd / abs(pmean) * 100.0) if abs(pmean) > 1e-12 else float("nan")
        txt2 = f"μ={pmean:.4g}  σ={pstd:.4g}  CV={pcv:.3g}%  n={pooled.size}"
        ax3.text(0.98, 0.98, txt2, transform=ax3.transAxes, va="top", ha="right",
                 fontsize=7, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#999", alpha=0.8))
    else:
        ax3.text(0.5, 0.5, "No pixel data.", ha="center", va="center"); ax3.axis("off")
    ax3.set_xlabel(clab); ax3.set_ylabel("density")
    ax3.set_title("Pooled value distribution (all files)", fontsize=9)
    ax3.grid(alpha=0.3)
    return fig


def grouped_line_table(scans: list, line_id: str, channel: str = "eyy",
                       label: str = "without_roi", *, width: int = 3):
    """DataFrame: per-file mean (and intra-line std) of the line + an
    'ALL (summary)' row with Mean/Std/SE/CV/95%CI across files."""
    import pandas as pd
    g = grouped_line_profiles(scans, line_id, channel, label, width=width)
    rows = [{"scan": n, "line_mean": round(m, 6), "line_std": round(sd, 6)}
            for n, m, sd in zip(g["names"], g["per_file_mean"], g["per_file_std"])]
    s = g["summary"]
    if s.get("n"):
        rows.append({"scan": "ALL (summary)", "line_mean": round(s["mean"], 6),
                     "std": round(s["std"], 6), "sem": round(s["sem"], 6),
                     "cv_%": round(s["cv_pct"], 3),
                     "ci95_lo": round(s["ci95_lo"], 6), "ci95_hi": round(s["ci95_hi"], 6),
                     "n": s["n"]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PIXEL-WISE DIFFERENCE — repeatability between repeated strain/stress maps:
#   Δε_yy(x,y) = ε_yy^B(x,y) − ε_yy^A(x,y)  →  MAE, RMSE, bias, std(Δ), correlation
# ─────────────────────────────────────────────────────────────────────────────

def channel_map_2d(scan: "Scan", channel: str = "eyy", label: str = "without_roi"):
    """A 2-D map for one channel: strain εyy/εxx/εxy in %, orientation θ in degrees,
    stress σxx/σyy/σxy in the display units, or the ADF. Returns None when unavailable."""
    from fast_artifacts import _as_hw3
    st = scan.ensure_state()
    ch = str(channel).lower()
    if ch in _CH_IDX:
        hw3 = _as_hw3((getattr(st, "strain_raw", {}) or {}).get(label))
        if hw3 is None:
            return None
        return np.asarray(hw3[..., _CH_IDX[ch]], dtype=float) * 100.0
    if ch in _ORIENT_CH:
        from pipeline import _strain_maps_dict_from_raw
        raw = (getattr(st, "strain_raw", {}) or {}).get(label)
        maps = _strain_maps_dict_from_raw(raw)
        th = maps.get("theta")
        if th is None:
            return None
        a = np.asarray(th, dtype=float)
        finite = a[np.isfinite(a)]
        if finite.size:
            mx = float(np.nanmax(np.abs(finite)))
            p99 = float(np.nanpercentile(np.abs(finite), 99))
            if mx <= float(np.pi) + 0.08 and p99 <= float(np.pi) + 0.08:
                a = np.rad2deg(a)
        return a
    if ch in _STRESS_KEY:
        stress = (getattr(st, "stress_tensors_pa", {}) or {}).get(label)
        if not stress:
            return None
        arr = stress.get(_STRESS_KEY[ch])
        if arr is None:
            return None
        return np.asarray(arr, dtype=float) / _stress_scale(scan)[0]
    if ch == "adf":
        return cached_adf(scan)
    return None


def pixel_difference(scan_a: "Scan", scan_b: "Scan", channel: str = "eyy",
                     label: str = "without_roi", *, drift_correct: bool = False) -> dict | None:
    """Δ(x,y) = map_B − map_A pixel-wise (A = reference / first repeat), with
    repeatability metrics: MAE = mean|Δ|, RMSE = √mean(Δ²), bias = mean(Δ),
    std(Δ), and Pearson correlation between A and B. If ``drift_correct`` and
    both scans have a loaded ``.drift``, B is sub-pixel shifted onto A's pixel
    frame (using the same convention as ``_line_drift_shift``) before Δ is
    computed. Returns
    ``{a, b, delta, metrics{mae,rmse,bias,std,corr,n}, drift_a, drift_b, drift_shift}``
    or None."""
    a = channel_map_2d(scan_a, channel, label)
    b = channel_map_2d(scan_b, channel, label)
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.shape != b.shape:                      # crop to the common region
        h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    drift_a, drift_b = getattr(scan_a, "drift", None), getattr(scan_b, "drift", None)
    drift_shift = None
    if drift_correct and drift_a and drift_b:
        from scipy.ndimage import shift as _ndi_shift
        drift_shift = (drift_b[1] - drift_a[1], drift_b[0] - drift_a[0])
        b = _ndi_shift(b, drift_shift, order=1, mode="constant", cval=np.nan)
    delta = b - a
    m = np.isfinite(delta) & np.isfinite(a) & np.isfinite(b)
    dv, av, bv = delta[m], a[m], b[m]
    n = int(dv.size)
    met = {
        "mae": float(np.mean(np.abs(dv))) if n else float("nan"),
        "rmse": float(np.sqrt(np.mean(dv ** 2))) if n else float("nan"),
        "bias": float(np.mean(dv)) if n else float("nan"),
        "std": float(np.std(dv, ddof=1)) if n > 1 else 0.0,
        "corr": float(np.corrcoef(av, bv)[0, 1]) if n > 1 else float("nan"),
        "n": n,
    }
    return {"a": a, "b": b, "delta": delta, "metrics": met,
            "drift_a": drift_a, "drift_b": drift_b, "drift_shift": drift_shift}


def build_pixel_difference_figure(scan_a: "Scan", scan_b: "Scan", channel: str = "eyy",
                                  label: str = "without_roi", *, drift_correct: bool = False):
    """A, B, Δ=B−A maps (shared symmetric scale) + an A-vs-B scatter with the
    repeatability metrics box (MAE/RMSE/bias/std(Δ)/corr)."""
    from matplotlib.figure import Figure
    res = pixel_difference(scan_a, scan_b, channel, label, drift_correct=drift_correct)
    fig = Figure(figsize=(12.4, 3.7), constrained_layout=True)
    if res is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data for both scans on this channel/map.\n"
                "(Need computed strain — and stress for σ channels.)",
                ha="center", va="center"); ax.axis("off")
        return fig
    a, b, delta, met = res["a"], res["b"], res["delta"], res["metrics"]
    clab = _channel_label(scan_a, channel)
    allv = np.concatenate([a[np.isfinite(a)].ravel(), b[np.isfinite(b)].ravel()])
    vmax = float(np.percentile(np.abs(allv), 99)) if allv.size else 1.0
    vmax = vmax or 1.0
    dfin = np.abs(delta[np.isfinite(delta)])
    dmax = float(np.percentile(dfin, 99)) if dfin.size else 1.0
    dmax = dmax or 1.0
    cmap = "RdBu_r" if channel != "adf" else "gray"
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(a, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper")
    ax1.set_title(f"A: {scan_a.name}\n{clab}", fontsize=8); ax1.axis("off")
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(b, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper")
    ax2.set_title(f"B: {scan_b.name}\n{clab}", fontsize=8); ax2.axis("off")
    ax3 = fig.add_subplot(1, 4, 3)
    im = ax3.imshow(delta, cmap="RdBu_r", vmin=-dmax, vmax=dmax, origin="upper")
    ax3.set_title("Δ = B − A", fontsize=8); ax3.axis("off")
    fig.colorbar(im, ax=ax3, fraction=0.046)
    ax4 = fig.add_subplot(1, 4, 4)
    mf = np.isfinite(a) & np.isfinite(b)
    av, bv = a[mf].ravel(), b[mf].ravel()
    step = max(1, av.size // 4000)
    ax4.scatter(av[::step], bv[::step], s=2, alpha=0.25, color="#1565C0")
    lim = [-vmax, vmax]; ax4.plot(lim, lim, "k--", lw=0.8)
    ax4.set_xlim(lim); ax4.set_ylim(lim)
    ax4.set_xlabel("A"); ax4.set_ylabel("B"); ax4.set_title("A vs B", fontsize=8)
    ax4.grid(alpha=0.3)
    txt = (f"MAE   = {met['mae']:.4g}\nRMSE  = {met['rmse']:.4g}\n"
           f"bias  = {met['bias']:+.4g}\nstd(Δ)= {met['std']:.4g}\n"
           f"corr  = {met['corr']:.4f}\nn     = {met['n']}")
    ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, va="top", ha="left",
             fontsize=7, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#999", alpha=0.85))
    title = f"Pixel-wise difference (repeatability) — {clab} ({label})"
    if res["drift_shift"] is not None:
        dx, dy = res["drift_shift"][1], res["drift_shift"][0]
        title += f"  ·  drift-corrected (Δ={dx:+.1f},{dy:+.1f} px)"
    elif drift_correct:
        title += "  ·  drift correction requested but drift not loaded for A/B"
    fig.suptitle(title, fontsize=10)
    return fig


def pixel_difference_table(scans: list, channel: str = "eyy", label: str = "without_roi",
                           *, reference_idx: int = 0, drift_correct: bool = False):
    """Repeatability metrics of every scan vs the reference scan → DataFrame
    (MAE, RMSE, bias, std(Δ), corr, n)."""
    import pandas as pd
    if not scans:
        return pd.DataFrame()
    ri = reference_idx if 0 <= reference_idx < len(scans) else 0
    ref = scans[ri]
    rows = []
    for i, sc in enumerate(scans):
        if i == ri:
            continue
        res = pixel_difference(ref, sc, channel, label, drift_correct=drift_correct)
        if res is None:
            continue
        m = res["metrics"]
        row = {"scan": sc.name, "ref": ref.name,
               "MAE": round(m["mae"], 6), "RMSE": round(m["rmse"], 6),
               "bias": round(m["bias"], 6), "std_delta": round(m["std"], 6),
               "corr": round(m["corr"], 6), "n": m["n"]}
        ref_drift, sc_drift = getattr(ref, "drift", None), getattr(sc, "drift", None)
        if ref_drift and sc_drift:
            row["drift_dx"] = round(sc_drift[0] - ref_drift[0], 3)
            row["drift_dy"] = round(sc_drift[1] - ref_drift[1], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def build_maps_with_lines_figure(scan: "Scan", label: str = "without_roi", *, width: int = 3):
    """A grid of EVERY available map (εyy/εxx/εxy + σxx/σyy/σxy when stress exists) with
    this scan's line segments overlaid on each — so you see εxx (and all maps) with its
    lines — and, below each map, that channel's line-profile plot (one curve per line),
    like ``build_line_profiles_figure`` but for every map panel. ADF is added as a
    reference panel too."""
    from matplotlib.figure import Figure
    panels = []
    for ch in ("eyy", "exx", "exy", "sxx", "syy", "sxy"):
        m = channel_map_2d(scan, ch, label)
        if m is not None and getattr(m, "ndim", 0) == 2:
            panels.append((ch, m, _channel_label(scan, ch), "RdBu_r", True))
    adf = cached_adf(scan)
    if adf is not None and getattr(adf, "ndim", 0) == 2:
        panels.append(("adf", np.asarray(adf, dtype=float), "ADF", "gray", False))
    lines = sorted((getattr(scan, "lines", None) or {}).items())
    prof = line_profiles(scan, label, width=width)
    if not panels:
        fig = Figure(figsize=(11.5, 7.2), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No maps for this scan/label.\n(Compute strain — and stress "
                "for σ channels.)", ha="center", va="center"); ax.axis("off")
        return fig
    import math
    ncol = 3
    nrow = max(1, math.ceil(len(panels) / ncol))
    fig = Figure(figsize=(12.5, 4.0 * nrow), constrained_layout=True)
    gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=[3, 1] * nrow)
    for i, (ch, m, lab, cmap, symmetric) in enumerate(panels):
        r, c = divmod(i, ncol)
        ax = fig.add_subplot(gs[2 * r, c])
        v = m[np.isfinite(m)]
        if symmetric:
            vmax = float(np.percentile(np.abs(v), 98)) if v.size else 1.0
            ax.imshow(m, cmap=cmap, vmin=-(vmax or 1.0), vmax=(vmax or 1.0), origin="upper")
        else:
            lo, hi = ((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                      if v.size else (float(m.min()), float(m.max() or 1)))
            ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi, origin="upper")
        for j, (_lid, seg) in enumerate(lines):
            try:
                (x0, y0), (x1, y1) = seg
            except Exception:
                continue
            col = SIX_POINT_COLORS[j % len(SIX_POINT_COLORS)]
            ax.plot([x0, x1], [y0, y1], color=col, lw=1.6)
        ax.set_title(lab, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

        ax_p = fig.add_subplot(gs[2 * r + 1, c])
        has_curve = False
        for j, (lid, _seg) in enumerate(lines):
            pr = prof.get(lid, {})
            if ch in pr and "dist" in pr:
                col = SIX_POINT_COLORS[j % len(SIX_POINT_COLORS)]
                ax_p.plot(pr["dist"], pr[ch], color=col, lw=1.0, label=lid)
                has_curve = True
        ax_p.grid(alpha=0.3)
        ax_p.set_xlabel("distance (px)", fontsize=7)
        ax_p.tick_params(labelsize=7)
        if i == 0 and has_curve:
            ax_p.legend(fontsize=6, ncol=min(4, len(lines)))
    fig.suptitle(f"{scan.name} — maps with lines + profiles ({label})", fontsize=10)
    return fig


def build_single_line_map_figure(scan: "Scan", line_id: str, channel: str = "eyy",
                                 label: str = "without_roi", *, width: int = 3):
    """One map of ``channel`` with ONLY ``line_id`` drawn on it, profile underneath.

    The focused single-line counterpart to ``build_maps_with_lines_figure`` — for
    when you want a clean figure of just one line on one map (e.g. for a paper)."""
    from matplotlib.figure import Figure
    m = channel_map_2d(scan, channel, label)
    seg = (getattr(scan, "lines", None) or {}).get(line_id)
    clab = _channel_label(scan, channel)
    fig = Figure(figsize=(6.4, 8.4), constrained_layout=True)
    if m is None or getattr(m, "ndim", 0) != 2 or seg is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"No map / line for {line_id} ({channel}).",
                ha="center", va="center"); ax.axis("off")
        return fig
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0, 0])
    v = m[np.isfinite(m)]
    symmetric = not str(channel).startswith("adf")
    if symmetric:
        vmax = float(np.percentile(np.abs(v), 98)) if v.size else 1.0
        ax.imshow(m, cmap="RdBu_r", vmin=-(vmax or 1.0), vmax=(vmax or 1.0), origin="upper")
    else:
        lo, hi = ((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                  if v.size else (float(m.min()), float(m.max() or 1)))
        ax.imshow(m, cmap="gray", vmin=lo, vmax=hi, origin="upper")
    try:
        (x0, y0), (x1, y1) = seg
        ax.plot([x0, x1], [y0, y1], color="#e6194b", lw=2.0)
    except Exception:
        pass
    ax.set_title(f"{scan.name} — {clab} with {line_id} ({label})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax_p = fig.add_subplot(gs[1, 0])
    pr = line_profiles(scan, label, width=width).get(line_id, {})
    if channel in pr and "dist" in pr:
        ax_p.plot(pr["dist"], pr[channel], color="#e6194b", lw=1.2)
    ax_p.set_xlabel("distance (px)", fontsize=8); ax_p.set_ylabel(clab, fontsize=8)
    ax_p.grid(alpha=0.3); ax_p.tick_params(labelsize=7)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# AREA-ROI ANALYSIS — same machinery as the lines, but each ROI is a rectangle:
#   per-file the statistic is the ROI mean (over all pixels inside) per channel;
#   the grouped/across-file view compares those per-file ROI means (mean±std, CV,
#   95% CI), exactly like the grouped line view. Mirrors the line pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def _roi_region_values(arr2d, bounds) -> "np.ndarray":
    """Flattened finite values of ``arr2d`` inside the ROI [x0,x1,y0,y1] (x=col,
    y=row), clamped to the array. Empty array when the ROI is degenerate/outside."""
    a = np.asarray(arr2d, dtype=float)
    if a.ndim != 2:
        return np.zeros(0)
    H, W = a.shape
    x0, x1, y0, y1 = (int(round(float(b))) for b in bounds)
    x0, x1 = sorted((max(0, min(x0, W - 1)), max(0, min(x1, W - 1))))
    y0, y1 = sorted((max(0, min(y0, H - 1)), max(0, min(y1, H - 1))))
    sub = a[y0:y1 + 1, x0:x1 + 1]
    v = sub[np.isfinite(sub)]
    return v.ravel()


def _roi_stat(values) -> dict:
    """{values, mean, std, n} for one ROI/channel sample (used by the figures/tables)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    return {"values": v,
            "mean": float(np.mean(v)) if n else float("nan"),
            "std": float(np.std(v, ddof=1)) if n > 1 else 0.0,
            "n": n}


def roi_profiles(scan: "Scan", label: str = "without_roi") -> dict:
    """{roi_id: {channel: {values, mean, std, n}}} — per-ROI pixel statistics from the
    saved strain map (``label``, channels in %), stress maps (display units) and ADF.
    The ROI analog of ``line_profiles`` (a rectangle has no distance axis, so each
    ROI reduces to a sample/mean per channel)."""
    from fast_artifacts import _as_hw3
    st = scan.ensure_state()
    hw3 = _as_hw3((getattr(st, "strain_raw", {}) or {}).get(label))
    stress = (getattr(st, "stress_tensors_pa", {}) or {}).get(label)
    sdiv, _u = _stress_scale(scan)
    adf = cached_adf(scan)
    out: dict = {}
    for rid, bounds in scan_area_rois(scan).items():
        d: dict = {}
        if hw3 is not None:
            for ch, idx in _CH_IDX.items():
                d[ch] = _roi_stat(_roi_region_values(hw3[..., idx], bounds) * 100.0)
        if stress:
            for ch, key in _STRESS_KEY.items():
                arr = stress.get(key)
                if arr is not None:
                    d[ch] = _roi_stat(_roi_region_values(np.asarray(arr, dtype=float), bounds) / sdiv)
        if adf is not None:
            d["adf"] = _roi_stat(_roi_region_values(adf, bounds))
        if d:
            out[rid] = d
    return out


def build_roi_profiles_figure(scan: "Scan", label: str = "without_roi",
                              channel: str = "eyy", *, register: bool = True):
    """Top: the chosen map with this scan's ROI rectangles overlaid. Bottom: a bar of
    each ROI's mean (±std) for that channel. ROI analog of ``build_line_profiles_figure``."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle as plt_Rectangle
    prof = roi_profiles(scan, label)
    base = channel_map_2d(scan, channel, label)
    ylab = _channel_label(scan, channel)
    fig = Figure(figsize=(7.6, 7.2), constrained_layout=True)
    ax_map = fig.add_subplot(2, 1, 1)
    if base is not None and getattr(base, "ndim", 0) == 2:
        if channel == "adf":
            v = base[base > 0]
            lo, hi = ((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                      if v.size else (float(base.min()), float(base.max() or 1)))
            ax_map.imshow(base, cmap="gray", vmin=lo, vmax=hi, origin="upper")
        else:
            m = float(np.nanpercentile(np.abs(base), 95)) or 1.0
            ax_map.imshow(base, cmap="RdBu_r", vmin=-m, vmax=m, origin="upper")
    ax_map.set_title(f"{scan.name} — {ylab} ({label})", fontsize=9)
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ids = sorted(scan_area_rois(scan))
    means, stds, errn = [], [], []
    for i, rid in enumerate(ids):
        c = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
        x0, x1, y0, y1 = scan_area_rois(scan)[rid]
        ax_map.add_patch(plt_Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=c, lw=1.6))
        ax_map.text(x0 + 2, y0 - 2, rid, color=c, fontsize=7, va="bottom")
        st = prof.get(rid, {}).get(channel, {})
        means.append(st.get("mean", float("nan")))
        stds.append(st.get("std", 0.0))
        errn.append(c)
    ax_b = fig.add_subplot(2, 1, 2)
    if ids:
        ax_b.bar(range(len(ids)), means, yerr=stds, color=errn, alpha=0.85,
                 capsize=3, error_kw={"elinewidth": 1})
        ax_b.set_xticks(range(len(ids))); ax_b.set_xticklabels(ids)
    ax_b.axhline(0, color="#999", lw=0.7, ls="--")
    ax_b.set_ylabel(ylab); ax_b.set_title("ROI means (±std)", fontsize=9)
    ax_b.grid(alpha=0.3, axis="y")
    if register:
        register_figure(scan, "roi_profiles", fig)
    return fig


def build_roi_distribution_figure(scan: "Scan", label: str = "without_roi",
                                  channel: str = "eyy"):
    """How CONSTANT each ROI is: the within-ROI distribution of pixel values.
    Top: the chosen map with the ROI rectangles. Bottom: a step histogram (density)
    per ROI on a shared axis + a μ/σ/CV/n box — a narrow peak ⇒ constant values, a
    wide spread ⇒ variable. CV (= σ/|μ|·100%) quantifies the constancy directly."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle as plt_Rectangle
    prof = roi_profiles(scan, label)
    base = channel_map_2d(scan, channel, label)
    ylab = _channel_label(scan, channel)
    rois = scan_area_rois(scan)
    fig = Figure(figsize=(8.4, 7.4), constrained_layout=True)
    ax_map = fig.add_subplot(2, 1, 1)
    if base is not None and getattr(base, "ndim", 0) == 2:
        if channel == "adf":
            v = base[base > 0]
            lo, hi = ((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                      if v.size else (float(base.min()), float(base.max() or 1)))
            ax_map.imshow(base, cmap="gray", vmin=lo, vmax=hi, origin="upper")
        else:
            mm = float(np.nanpercentile(np.abs(base), 95)) or 1.0
            ax_map.imshow(base, cmap="RdBu_r", vmin=-mm, vmax=mm, origin="upper")
    for i, rid in enumerate(sorted(rois)):
        c = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
        x0, x1, y0, y1 = rois[rid]
        ax_map.add_patch(plt_Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=c, lw=1.6))
        ax_map.text(x0 + 2, y0 - 2, rid, color=c, fontsize=7, va="bottom")
    ax_map.set_title(f"{scan.name} — {ylab} ({label})", fontsize=9)
    ax_map.set_xticks([]); ax_map.set_yticks([])

    ax_h = fig.add_subplot(2, 1, 2)
    per = {}
    allvals = []
    for rid in sorted(rois):
        v = np.asarray(prof.get(rid, {}).get(channel, {}).get("values", []), dtype=float)
        v = v[np.isfinite(v)]
        per[rid] = v
        if v.size:
            allvals.append(v)
    if allvals:
        cat = np.concatenate(allvals)
        lo, hi = (float(np.percentile(cat, 0.5)), float(np.percentile(cat, 99.5)))
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            lo, hi = float(cat.min()), float(cat.max() or (lo + 1.0))
        bins = np.linspace(lo, hi, 41)
        txt = []
        for i, rid in enumerate(sorted(rois)):
            v = per[rid]
            if not v.size:
                continue
            c = SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)]
            ax_h.hist(v, bins=bins, histtype="step", density=True, color=c, lw=1.6, label=rid)
            mean = float(np.mean(v))
            std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
            ax_h.axvline(mean, color=c, lw=1.0, ls="--", alpha=0.8)
            cv = (std / abs(mean) * 100.0) if abs(mean) > 1e-12 else float("nan")
            txt.append(f"{rid}: μ={mean:.3g}  σ={std:.3g}  CV={cv:.2g}%  n={v.size}")
        ax_h.legend(fontsize=7, ncol=min(4, len(per)))
        ax_h.text(0.98, 0.98, "\n".join(txt), transform=ax_h.transAxes, va="top", ha="right",
                  fontsize=7, family="monospace",
                  bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#999", alpha=0.8))
    else:
        ax_h.text(0.5, 0.5, "No ROI data — set ROIs + compute strain first.",
                  ha="center", va="center"); ax_h.axis("off")
    ax_h.set_xlabel(ylab); ax_h.set_ylabel("density")
    ax_h.set_title("Within-ROI value distribution  (narrow ⇒ constant · CV = σ/|μ|)", fontsize=9)
    ax_h.grid(alpha=0.3)
    register_figure(scan, "roi_distribution", fig)
    return fig


def grouped_roi_profiles(scans: list, roi_id: str, channel: str = "eyy",
                         label: str = "without_roi") -> dict:
    """The SAME ROI across files. Returns {names, per_file_mean, per_file_std, pooled,
    summary{mean,std,sem,cv_pct,ci95_lo,ci95_hi,n,mean_intra_std,cv_intra_pct,cv_ratio}}
    — summary stats are across the per-file ROI means (mirrors
    ``grouped_line_profiles``); per_file_std is each file's intra-ROI pixel std and
    pooled concatenates every file's ROI pixels (run-level spread).

    ``cv_pct`` is the BETWEEN-file CV (run-to-run reproducibility); ``cv_intra_pct``
    is the average WITHIN-ROI CV (sample heterogeneity, from ``per_file_std``).
    ``cv_ratio = cv_pct / cv_intra_pct`` is >1 when run-to-run disagreement exceeds
    the ROI's own intrinsic spread (reproducibility-limited) and <1 when the ROI is
    simply heterogeneous but the repeats agree well (ROI-heterogeneity-limited)."""
    names, per, per_std, pooled = [], [], [], []
    for sc in scans:
        st = roi_profiles(sc, label).get(roi_id, {}).get(channel)
        if st and st.get("n"):
            names.append(sc.name)
            per.append(st["mean"])
            per_std.append(st.get("std", 0.0))
            pooled.append(np.asarray(st["values"], dtype=float))
    res: dict = {"names": names, "per_file_mean": per, "per_file_std": per_std,
                 "pooled": (np.concatenate(pooled) if pooled else np.zeros(0))}
    res["summary"] = _summary_stats(np.asarray(per)) if per else {}
    s = res["summary"]
    if s.get("n"):
        mean_intra = float(np.mean(per_std)) if per_std else float("nan")
        cv_intra = (mean_intra / abs(s["mean"]) * 100.0
                    if abs(s["mean"]) > 1e-12 else float("nan"))
        s["mean_intra_std"] = mean_intra
        s["cv_intra_pct"] = cv_intra
        s["cv_ratio"] = (s["cv_pct"] / cv_intra
                         if np.isfinite(cv_intra) and abs(cv_intra) > 1e-12
                         else float("nan"))
    return res


def build_grouped_roi_figure(scans: list, roi_id: str, channel: str = "eyy",
                             label: str = "without_roi"):
    """Left: per-file ROI mean bars (±intra-ROI std error bars) + the across-files
    mean±std band and summary stats box. Right: pooled (all files' ROI pixels)
    value histogram — the "run-level" spread. Mirrors
    ``build_grouped_line_figure``."""
    from matplotlib.figure import Figure
    g = grouped_roi_profiles(scans, roi_id, channel, label)
    fig = Figure(figsize=(6.8, 9.0), constrained_layout=True)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    names, per, per_std = g["names"], g["per_file_mean"], g["per_file_std"]
    clab = _channel_label(scans[0], channel) if scans else _CH_LABEL.get(channel, channel)
    if not names:
        ax1.text(0.5, 0.5, "No ROI data — set ROIs + compute strain first.",
                 ha="center", va="center"); ax1.axis("off"); ax2.axis("off")
        return fig
    s = g["summary"]
    mean = s.get("mean", float(np.mean(per)))
    std = s.get("std", 0.0)
    cols = [SIX_POINT_COLORS[i % len(SIX_POINT_COLORS)] for i in range(len(names))]
    xs = range(len(names))
    ax1.bar(xs, per, yerr=per_std, color=cols, alpha=0.85,
            capsize=3, error_kw={"elinewidth": 1})
    ax1.axhline(mean, color="black", lw=2.0, label="mean")
    ax1.axhspan(mean - std, mean + std, color="0.5", alpha=0.2, label="±std")
    ax1.set_xticks(list(xs)); ax1.set_xticklabels([n[:14] for n in names], rotation=30, ha="right")
    ax1.set_ylabel(clab)
    ax1.set_title(f"{roi_id} across {len(names)} file(s) — {clab} ({label})", fontsize=9)
    ax1.grid(alpha=0.3, axis="y"); ax1.legend(fontsize=7)
    if s.get("n"):
        ratio = s.get("cv_ratio", float("nan"))
        if np.isfinite(ratio):
            qual = "reproducibility-limited" if ratio > 1 else "ROI-heterogeneity-limited"
            ratio_txt = f"ratio={ratio:.2g}  ({qual})"
        else:
            ratio_txt = "ratio=n/a"
        txt = (f"across-file mean of ROI means:\nMean={s['mean']:.4g}  Std={s['std']:.4g}\n"
               f"SE={s['sem']:.4g}  CV_between={s['cv_pct']:.3g}%\n"
               f"95% CI=[{s['ci95_lo']:.4g}, {s['ci95_hi']:.4g}]  n={s['n']}\n"
               f"CV_within(ROI)={s['cv_intra_pct']:.3g}%  {ratio_txt}")
        ax1.text(0.02, 0.98, txt, transform=ax1.transAxes, va="top", ha="left", fontsize=7,
                 bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#999", alpha=0.7))

    pooled = g["pooled"]
    if pooled.size:
        lo, hi = (float(np.percentile(pooled, 0.5)), float(np.percentile(pooled, 99.5)))
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            lo, hi = float(pooled.min()), float(pooled.max() or (lo + 1.0))
        bins = np.linspace(lo, hi, 41)
        ax2.hist(pooled, bins=bins, histtype="step", density=True, color="#3cb44b", lw=1.6)
        pmean = float(np.mean(pooled))
        pstd = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0
        ax2.axvline(pmean, color="#3cb44b", lw=1.0, ls="--", alpha=0.8)
        pcv = (pstd / abs(pmean) * 100.0) if abs(pmean) > 1e-12 else float("nan")
        txt2 = f"μ={pmean:.4g}  σ={pstd:.4g}  CV={pcv:.3g}%  n={pooled.size}"
        ax2.text(0.98, 0.98, txt2, transform=ax2.transAxes, va="top", ha="right",
                 fontsize=7, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#999", alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "No pixel data.", ha="center", va="center"); ax2.axis("off")
    ax2.set_xlabel(clab); ax2.set_ylabel("density")
    ax2.set_title("Pooled pixel distribution (all files)", fontsize=9)
    ax2.grid(alpha=0.3)
    return fig


def grouped_roi_table(scans: list, roi_id: str, channel: str = "eyy",
                      label: str = "without_roi"):
    """DataFrame: per-file ROI mean (and intra-ROI std) + an 'ALL (summary)' row with
    Mean/Std/SE/CV_between/95%CI across files plus CV_within (ROI heterogeneity, from
    per-file intra-ROI std) and their ratio. Mirrors ``grouped_line_table``."""
    import pandas as pd
    g = grouped_roi_profiles(scans, roi_id, channel, label)
    rows = [{"scan": n, "roi_mean": round(m, 6), "roi_std": round(sd, 6)}
            for n, m, sd in zip(g["names"], g["per_file_mean"], g["per_file_std"])]
    s = g["summary"]
    if s.get("n"):
        rows.append({"scan": "ALL (summary)", "roi_mean": round(s["mean"], 6),
                     "std": round(s["std"], 6), "sem": round(s["sem"], 6),
                     "cv_between_%": round(s["cv_pct"], 3),
                     "cv_within_%": round(s["cv_intra_pct"], 3),
                     "cv_ratio": round(s["cv_ratio"], 3) if np.isfinite(s["cv_ratio"]) else None,
                     "ci95_lo": round(s["ci95_lo"], 6), "ci95_hi": round(s["ci95_hi"], 6),
                     "n": s["n"]})
    return pd.DataFrame(rows)


def build_maps_with_rois_figure(scan: "Scan", label: str = "without_roi"):
    """Every available map (εyy/εxx/εxy + σxx/σyy/σxy when stress exists) + ADF, each
    with this scan's ROI rectangles overlaid. Mirrors ``build_maps_with_lines_figure``."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle as plt_Rectangle
    panels = []
    for ch in ("eyy", "exx", "exy", "sxx", "syy", "sxy"):
        m = channel_map_2d(scan, ch, label)
        if m is not None and getattr(m, "ndim", 0) == 2:
            panels.append((ch, m, _channel_label(scan, ch), "RdBu_r", True))
    adf = cached_adf(scan)
    if adf is not None and getattr(adf, "ndim", 0) == 2:
        panels.append(("adf", np.asarray(adf, dtype=float), "ADF", "gray", False))
    rois = scan_area_rois(scan)
    fig = Figure(figsize=(11.5, 7.2), constrained_layout=True)
    if not panels:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No maps for this scan/label.\n(Compute strain — and stress "
                "for σ channels.)", ha="center", va="center"); ax.axis("off")
        return fig
    import math
    ncol = 3
    nrow = max(1, math.ceil(len(panels) / ncol))
    for i, (ch, m, lab, cmap, symmetric) in enumerate(panels):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        v = m[np.isfinite(m)]
        if symmetric:
            vmax = float(np.percentile(np.abs(v), 98)) if v.size else 1.0
            ax.imshow(m, cmap=cmap, vmin=-(vmax or 1.0), vmax=(vmax or 1.0), origin="upper")
        else:
            lo, hi = ((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                      if v.size else (float(m.min()), float(m.max() or 1)))
            ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi, origin="upper")
        for j, rid in enumerate(sorted(rois)):
            x0, x1, y0, y1 = rois[rid]
            c = SIX_POINT_COLORS[j % len(SIX_POINT_COLORS)]
            ax.add_patch(plt_Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=c, lw=1.6))
        ax.set_title(lab, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{scan.name} — maps with ROIs ({label})", fontsize=10)
    return fig


def build_single_roi_map_figure(scan: "Scan", roi_id: str, channel: str = "eyy",
                                label: str = "without_roi"):
    """One map of ``channel`` with ONLY ``roi_id`` drawn on it — the focused
    single-ROI counterpart to ``build_maps_with_rois_figure``."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle as plt_Rectangle
    m = channel_map_2d(scan, channel, label)
    bounds = scan_area_rois(scan).get(roi_id)
    clab = _channel_label(scan, channel)
    fig = Figure(figsize=(6.4, 6.2), constrained_layout=True)
    ax = fig.add_subplot(111)
    if m is None or getattr(m, "ndim", 0) != 2 or not bounds:
        ax.text(0.5, 0.5, f"No map / ROI for {roi_id} ({channel}).",
                ha="center", va="center"); ax.axis("off")
        return fig
    v = m[np.isfinite(m)]
    symmetric = not str(channel).startswith("adf")
    if symmetric:
        vmax = float(np.percentile(np.abs(v), 98)) if v.size else 1.0
        ax.imshow(m, cmap="RdBu_r", vmin=-(vmax or 1.0), vmax=(vmax or 1.0), origin="upper")
    else:
        lo, hi = ((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                  if v.size else (float(m.min()), float(m.max() or 1)))
        ax.imshow(m, cmap="gray", vmin=lo, vmax=hi, origin="upper")
    x0, x1, y0, y1 = bounds
    ax.add_patch(plt_Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor="#e6194b", lw=2.0))
    ax.set_title(f"{scan.name} — {clab} with {roi_id} ({label})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return fig


def register_live_roi_report_figures(scans: list, roi_ids: list, *,
                                     channel: str = "eyy", label: str = "without_roi",
                                     log: Log = None) -> list[str]:
    """Materialize Live-ROI figures only under ``report_*`` keys (Send to Report).

    Does **not** register generic ``roi_profiles`` / ``maps_with_rois``.
    Returns keys written on the first scan (for UI jump).
    """
    keys_written: list[str] = []
    for sc in scans:
        if not scan_area_rois(sc):
            continue
        for rid in roi_ids:
            key = f"report_roi_{rid}_{channel}_{label}"
            fig = build_single_roi_map_figure(sc, rid, channel, label)
            if register_figure(sc, key, fig, force=True):
                if sc is scans[0]:
                    keys_written.append(key)
        prof_key = f"report_roi_profiles_{channel}_{label}"
        fig = build_roi_profiles_figure(sc, label, channel, register=False)
        if register_figure(sc, prof_key, fig, force=True):
            if sc is scans[0] and prof_key not in keys_written:
                keys_written.append(prof_key)
    for rid in roi_ids:
        if not scans:
            break
        gkey = f"report_roi_group_{rid}_{channel}_{label}"
        fig = build_grouped_roi_figure(scans, rid, channel, label)
        if register_figure(scans[0], gkey, fig, force=True):
            keys_written.append(gkey)
    _log(log, f"Report (Send): {len(roi_ids)} ROI(s) on {len(scans)} file(s) → "
              f"{', '.join(keys_written) or '(none)'}")
    return keys_written


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION SUMMARY — fitted values / residual metrics across files
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATION_STEPS = ("origin", "ellipse", "q_pixel", "basis")
CALIBRATION_STEP_LABEL = {
    "origin": "Origin",
    "ellipse": "Ellipse",
    "q_pixel": "Q-pixel",
    "basis": "Basis",
}


def _finite_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else float("nan")
    except Exception:
        return float("nan")


def _origin_residual_metrics(scan: "Scan") -> dict:
    """Numeric comparison of origin residuals between files.

    py4DSTEM returns qx/qy origin residual maps after the origin fit. Useful
    cross-file scalars are RMSE, mean absolute residual, standard deviation,
    max absolute residual, and vector RMSE.
    """
    st = scan.ensure_state()
    fit = getattr(st, "origin_fit", None)
    if not (isinstance(fit, (list, tuple)) and len(fit) >= 4):
        return {}
    try:
        qx = np.asarray(fit[2], dtype=float)
        qy = np.asarray(fit[3], dtype=float)
    except Exception:
        return {}
    qx = qx[np.isfinite(qx)]
    qy = qy[np.isfinite(qy)]
    out: dict = {}
    if qx.size:
        out.update({
            "qx_residual_mean_px": float(np.mean(qx)),
            "qx_residual_mae_px": float(np.mean(np.abs(qx))),
            "qx_residual_rmse_px": float(np.sqrt(np.mean(qx ** 2))),
            "qx_residual_std_px": float(np.std(qx, ddof=1)) if qx.size > 1 else 0.0,
            "qx_residual_max_abs_px": float(np.max(np.abs(qx))),
        })
    if qy.size:
        out.update({
            "qy_residual_mean_px": float(np.mean(qy)),
            "qy_residual_mae_px": float(np.mean(np.abs(qy))),
            "qy_residual_rmse_px": float(np.sqrt(np.mean(qy ** 2))),
            "qy_residual_std_px": float(np.std(qy, ddof=1)) if qy.size > 1 else 0.0,
            "qy_residual_max_abs_px": float(np.max(np.abs(qy))),
        })
    if qx.size and qy.size:
        n = min(qx.size, qy.size)
        mag2 = qx[:n] ** 2 + qy[:n] ** 2
        out["origin_vector_rmse_px"] = float(np.sqrt(np.mean(mag2)))
        out["origin_vector_mean_abs_px"] = float(np.mean(np.sqrt(mag2)))
        out["origin_residual_n"] = int(n)
    return out


def _scan_calibration_metrics(scan: "Scan") -> dict:
    st = scan.ensure_state()
    p = scan.params
    row = {
        "scan": scan.name,
        "raw_path": scan.raw_path,
        "h5_path": scan_h5_path(scan),
        "braggpeaks_path": scan.braggpeaks_path,
    }
    row.update(_origin_residual_metrics(scan))

    pe = getattr(st, "p_ellipse", None)
    if isinstance(pe, (list, tuple)) and len(pe) >= 5:
        row.update({
            "ellipse_y0_px": _finite_float(pe[0]),
            "ellipse_x0_px": _finite_float(pe[1]),
            "ellipse_a_px": _finite_float(pe[2]),
            "ellipse_b_px": _finite_float(pe[3]),
            "ellipse_theta_rad": _finite_float(pe[4]),
            "ellipse_theta_deg": float(np.degrees(_finite_float(pe[4]))),
        })
        a, b = row["ellipse_a_px"], row["ellipse_b_px"]
        if np.isfinite(a) and np.isfinite(b) and b != 0:
            row["ellipse_a_over_b"] = float(a / b)
        if np.isfinite(a) and np.isfinite(b):
            row["ellipse_ab_delta_px"] = float(a - b)

    q_fit = getattr(p, "q_px_fitted", None)
    if q_fit is None:
        q_fit = getattr(st, "q_pixel_size", None)
    row.update({
        "q_pixel_guess_Ainv_per_px": _finite_float(getattr(p, "q_px", None)),
        "q_pixel_fitted_Ainv_per_px": _finite_float(q_fit),
    })
    qg, qf = row["q_pixel_guess_Ainv_per_px"], row["q_pixel_fitted_Ainv_per_px"]
    if np.isfinite(qg) and np.isfinite(qf):
        row["q_pixel_delta_Ainv_per_px"] = float(qf - qg)
        row["q_pixel_delta_pct"] = float((qf - qg) / qg * 100.0) if qg else float("nan")

    bp = getattr(st, "strain_basis_params", None) or {}
    g1 = bp.get("g1_qxy")
    g2 = bp.get("g2_qxy")
    if isinstance(g1, (list, tuple)) and len(g1) >= 2:
        row["basis_g1_qx"] = _finite_float(g1[0])
        row["basis_g1_qy"] = _finite_float(g1[1])
        row["basis_g1_norm"] = float(np.hypot(row["basis_g1_qx"], row["basis_g1_qy"]))
    if isinstance(g2, (list, tuple)) and len(g2) >= 2:
        row["basis_g2_qx"] = _finite_float(g2[0])
        row["basis_g2_qy"] = _finite_float(g2[1])
        row["basis_g2_norm"] = float(np.hypot(row["basis_g2_qx"], row["basis_g2_qy"]))
    if all(k in row and np.isfinite(row[k]) for k in
           ("basis_g1_qx", "basis_g1_qy", "basis_g2_qx", "basis_g2_qy")):
        dot = row["basis_g1_qx"] * row["basis_g2_qx"] + row["basis_g1_qy"] * row["basis_g2_qy"]
        den = row["basis_g1_norm"] * row["basis_g2_norm"]
        row["basis_angle_deg"] = float(np.degrees(np.arccos(np.clip(dot / den, -1, 1)))) if den else float("nan")
    row["basis_qr_rotation_deg"] = _finite_float(getattr(p, "qr_rotation", None))
    row["basis_qr_flip"] = bool(getattr(p, "qr_flip", False))
    return row


def calibration_values_table(scans: list):
    """One row per file with fitted calibration values and origin residual metrics."""
    import pandas as pd
    rows = [_scan_calibration_metrics(sc) for sc in scans]
    return pd.DataFrame(rows)


CALIBRATION_VALUE_GROUPS = {
    "origin": [
        "origin_vector_rmse_px", "origin_vector_mean_abs_px",
        "qx_residual_rmse_px", "qy_residual_rmse_px",
        "qx_residual_mae_px", "qy_residual_mae_px",
    ],
    "ellipse": [
        "ellipse_a_px", "ellipse_b_px", "ellipse_a_over_b",
        "ellipse_ab_delta_px", "ellipse_theta_deg",
    ],
    "q_pixel": [
        "q_pixel_guess_Ainv_per_px", "q_pixel_fitted_Ainv_per_px",
        "q_pixel_delta_Ainv_per_px", "q_pixel_delta_pct",
    ],
    "basis": [
        "basis_g1_qx", "basis_g1_qy", "basis_g1_norm",
        "basis_g2_qx", "basis_g2_qy", "basis_g2_norm",
        "basis_angle_deg", "basis_qr_rotation_deg",
    ],
}


def calibration_numeric_columns(scans: list, *, step: str | None = None) -> list:
    df = calibration_values_table(scans)
    if df.empty:
        return []
    cols = []
    preferred = CALIBRATION_VALUE_GROUPS.get(step or "", [])
    pool = preferred or [c for c in df.columns if c not in ("scan", "raw_path", "h5_path", "braggpeaks_path")]
    for c in pool:
        if c in df.columns:
            vals = np.asarray(df[c], dtype=float) if df[c].dtype != object else None
            if vals is not None and np.isfinite(vals).any():
                cols.append(c)
    return cols


def build_calibration_value_figure(scans: list, value: str | None = None):
    """Plot one fitted calibration value vs file index/name."""
    from matplotlib.figure import Figure
    df = calibration_values_table(scans)
    fig = Figure(figsize=(10.2, 4.8), constrained_layout=True)
    ax = fig.add_subplot(111)
    if df.empty:
        ax.text(0.5, 0.5, "No calibration data available.", ha="center", va="center")
        ax.axis("off")
        return fig
    cols = calibration_numeric_columns(scans)
    value = value if value in cols else (cols[0] if cols else None)
    if value is None:
        ax.text(0.5, 0.5, "No numeric calibration values available.\n"
                "Origin residual arrays are only available in this live session unless saved.",
                ha="center", va="center")
        ax.axis("off")
        return fig
    y = np.asarray(df[value], dtype=float)
    x = np.arange(len(df))
    ax.plot(x, y, "o-", color="#1565C0", lw=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in df["scan"]], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel(value)
    ax.set_xlabel("file")
    ax.set_title(f"Calibration value across files — {value}")
    ax.grid(alpha=0.3)
    finite = y[np.isfinite(y)]
    if finite.size:
        txt = (f"mean={np.mean(finite):.6g}\nstd={np.std(finite, ddof=1) if finite.size > 1 else 0:.6g}\n"
               f"min={np.min(finite):.6g}\nmax={np.max(finite):.6g}\nn={finite.size}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#999", alpha=0.85))
    return fig


def save_calibration_summary(scans: list, out_dir: str | "Path", *, log: Log = None) -> dict:
    """Write summary/calibrations/<step>/ with per-scan figures, CSVs and value plots."""
    out = Path(out_dir) / "calibrations"
    out.mkdir(parents=True, exist_ok=True)
    df = calibration_values_table(scans)
    n_plots = 0
    n_figs = 0
    if not df.empty:
        df.to_csv(out / "calibration_values_all.csv", index=False)
    for step in CALIBRATION_STEPS:
        step_dir = out / step
        step_dir.mkdir(parents=True, exist_ok=True)
        cols = [c for c in CALIBRATION_VALUE_GROUPS.get(step, []) if c in df.columns]
        keep = ["scan"] + cols
        if cols:
            df[keep].to_csv(step_dir / f"{step}_values.csv", index=False)
            for col in cols:
                try:
                    vals = np.asarray(df[col], dtype=float)
                    if not np.isfinite(vals).any():
                        continue
                    fig = build_calibration_value_figure(scans, col)
                    fig.savefig(step_dir / f"{col}.png", dpi=150, bbox_inches="tight")
                    n_plots += 1
                except Exception as exc:
                    _log(log, f"calibration plot {step}/{col} skipped: {exc}")
        for sc in scans:
            try:
                fig = collect_figures(sc).get(step)
                if fig is None:
                    continue
                safe = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in sc.name).strip()
                fig.savefig(step_dir / f"{safe}_{step}.png", dpi=150, bbox_inches="tight")
                n_figs += 1
            except Exception as exc:
                _log(log, f"calibration image {step}/{sc.name} skipped: {exc}")
    _log(log, f"Calibration summary written → {out} ({n_plots} plot(s), {n_figs} image(s)).")
    return {"plots": n_plots, "images": n_figs, "rows": int(len(df))}


# strain (always) + stress (when computed) channels for the grouped summary export
SUMMARY_CHANNELS = ("eyy", "exx", "exy", "sxx", "syy", "sxy")


def save_summary(scans: list, out_dir: str, *, width: int = 3, log: Log = None,
                 include_line_figs: bool = True, include_roi_figs: bool = True,
                 include_repeatability_figs: bool = True) -> dict:
    """Write a 'summary' folder: the SAME line/ROI across all files, grouped + summarized
    over every strain (εyy/εxx/εxy) and stress (σxx/σyy/σxy) map, as OriginLab-friendly
    CSVs (comma sep, dot decimal) + the grouped figures + a cross-file stats table.

    Layout::
        summary/
          summary_stats.csv                # one row per (kind, id, map, channel): Mean/Std/SE/CV/95%CI
          lines/<L>/<label>_<channel>_profiles.csv   # distance + per-file curves + mean + std
          lines/<L>/<label>_<channel>_table.csv      # per-file line mean/std + ALL(summary)
          lines/<L>/<label>_<channel>.png            # overlay + per-file mean±std + pooled histogram
          rois/<R>/<label>_<channel>_table.csv       # per-file ROI mean/std + ALL(summary)
          rois/<R>/<label>_<channel>.png             # per-file bars±std + pooled histogram

    CSV/table data is always written; the ``.png`` figures are gated by
    ``include_line_figs`` / ``include_roi_figs`` / ``include_repeatability_figs``
    (only the rendered images are skipped, never the underlying data).
    """
    import pandas as pd
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cal_res = save_calibration_summary(scans, out, log=log)
    line_ids = sorted({lid for sc in scans for lid in (getattr(sc, "lines", None) or {})})
    roi_ids = sorted({rid for sc in scans for rid in scan_area_rois(sc)})
    labels = [lab for lab in ("without_roi", "with_roi")
              if any((getattr(sc.state, "strain_raw", None) or {}).get(lab) is not None
                     for sc in scans if sc.state is not None)]
    if not line_ids and not roi_ids:
        _log(log, "Summary: no lines or ROIs set on any file — nothing to summarize.")
        return {"profiles": 0, "lines": 0, "rois": 0, "calibrations": cal_res}
    if not labels:
        labels = ["without_roi"]
    summary_rows: list = []
    n = 0
    for lid in line_ids:
        for lab in labels:
            for ch in SUMMARY_CHANNELS:
                g = grouped_line_profiles(scans, lid, ch, lab, width=width)
                if not g.get("series"):
                    continue
                d = out / "lines" / str(lid)
                d.mkdir(parents=True, exist_ok=True)
                cd = g["common_dist"]; m = len(cd)
                data = {"distance_px": np.asarray(cd)}
                for name, _dd, v in g["series"]:
                    data[str(name)] = np.asarray(v)[:m]
                data["mean"] = g["mean_curve"]; data["std"] = g["std_curve"]
                pd.DataFrame(data).to_csv(d / f"{lab}_{ch}_profiles.csv", index=False)
                try:
                    grouped_line_table(scans, lid, ch, lab, width=width).to_csv(
                        d / f"{lab}_{ch}_table.csv", index=False)
                except Exception as exc:
                    _log(log, f"summary table {lid}/{lab}/{ch} skipped: {exc}")
                if include_line_figs:
                    try:
                        fig = build_grouped_line_figure(scans, lid, ch, lab, width=width)
                        fig.savefig(d / f"{lab}_{ch}.png", dpi=_figure_policy.save_dpi,
                                    bbox_inches="tight")
                    except Exception as exc:
                        _log(log, f"summary figure {lid}/{lab}/{ch} skipped: {exc}")
                s = g.get("summary") or {}
                if s.get("n"):
                    summary_rows.append({
                        "kind": "line", "id": lid, "map": lab, "channel": ch,
                        "mean": s["mean"], "std": s["std"], "sem": s["sem"],
                        "cv_pct": s["cv_pct"], "ci95_lo": s["ci95_lo"],
                        "ci95_hi": s["ci95_hi"], "n": s["n"]})
                n += 1
    n_roi = 0
    for rid in roi_ids:
        for lab in labels:
            for ch in SUMMARY_CHANNELS:
                g = grouped_roi_profiles(scans, rid, ch, lab)
                if not g.get("names"):
                    continue
                d = out / "rois" / str(rid)
                d.mkdir(parents=True, exist_ok=True)
                try:
                    grouped_roi_table(scans, rid, ch, lab).to_csv(
                        d / f"{lab}_{ch}_table.csv", index=False)
                except Exception as exc:
                    _log(log, f"summary table {rid}/{lab}/{ch} skipped: {exc}")
                if include_roi_figs:
                    try:
                        fig = build_grouped_roi_figure(scans, rid, ch, lab)
                        fig.savefig(d / f"{lab}_{ch}.png", dpi=_figure_policy.save_dpi,
                                    bbox_inches="tight")
                    except Exception as exc:
                        _log(log, f"summary figure {rid}/{lab}/{ch} skipped: {exc}")
                s = g.get("summary") or {}
                if s.get("n"):
                    summary_rows.append({
                        "kind": "roi", "id": rid, "map": lab, "channel": ch,
                        "mean": s["mean"], "std": s["std"], "sem": s["sem"],
                        "cv_pct": s["cv_pct"], "ci95_lo": s["ci95_lo"],
                        "ci95_hi": s["ci95_hi"], "n": s["n"],
                        "cv_intra_pct": s["cv_intra_pct"], "cv_ratio": s["cv_ratio"]})
                n_roi += 1
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out / "summary_stats.csv", index=False)
    # ── pixel-wise repeatability (Δ = each scan − the first; MAE/RMSE/bias/std/corr) ──
    n_rep = 0
    if len(scans) >= 2:
        rep = out / "repeatability"
        for lab in labels:
            for ch in SUMMARY_CHANNELS:
                tbl = pixel_difference_table(scans, ch, lab, reference_idx=0,
                                             drift_correct=True)
                if tbl is None or tbl.empty:
                    continue
                rep.mkdir(parents=True, exist_ok=True)
                tbl.to_csv(rep / f"{lab}_{ch}_repeatability.csv", index=False)
                if include_repeatability_figs:
                    for j in range(1, len(scans)):    # Δ figure per repeat vs scan[0]
                        try:
                            if pixel_difference(scans[0], scans[j], ch, lab,
                                                drift_correct=True) is None:
                                continue
                            fig = build_pixel_difference_figure(scans[0], scans[j], ch, lab,
                                                                drift_correct=True)
                            fig.savefig(rep / f"{lab}_{ch}_{scans[j].name}_vs_{scans[0].name}.png",
                                        dpi=_figure_policy.save_dpi, bbox_inches="tight")
                        except Exception as exc:
                            _log(log, f"repeatability fig {ch}/{lab}/{j} skipped: {exc}")
                n_rep += 1
    _log(log, f"Summary written → {out} ({n} grouped line profile(s) over {len(line_ids)} "
              f"line(s) + {n_roi} ROI summary(ies) over {len(roi_ids)} ROI(s) "
              f"× {len(labels)} map(s); {n_rep} repeatability table(s)).")
    return {"profiles": n, "lines": len(line_ids), "rois": len(roi_ids),
            "roi_profiles": n_roi, "repeatability": n_rep, "calibrations": cal_res}


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE  (Parametros_cal.json) — fills calibration fields for single or multi
# ─────────────────────────────────────────────────────────────────────────────

def load_template(json_path: str):
    """Parse Parametros_cal.json → fast_batch.FastBatchConfig."""
    return _batch().load_fast_batch_config(json_path)


def params_from_template_scan(sc_cfg) -> CalibrationParams:
    """Translate one FastBatchScanConfig → CalibrationParams (fills GUI fields)."""
    s11 = sc_cfg.step11 or {}
    s12 = sc_cfg.step12 or {}
    s13 = sc_cfg.step13 or {}
    cbv = s12.get("choose_basis_vectors", {}) or {}
    vis = cbv.get("vis_params", {}) or {}
    return CalibrationParams(
        roi_bounds=list(sc_cfg.roi_bounds) if sc_cfg.roi_bounds else [],
        center_guess=list(sc_cfg.center_guess) if sc_cfg.center_guess else [128.0, 128.0],
        origin_sampling=int((sc_cfg.step10 or {}).get("sampling", 2)),
        q_px=float(s11.get("px_guess", 0.0137)),
        q_kmax=float(s11.get("kmax", 1.0)),
        q_kpow=float(s11.get("kpow", 2.0)),
        q_use_roi=bool(s11.get("use_roi", False)),
        qr_rotation=float(s12.get("qr_rotation", 0.0)),
        qr_flip=bool(s12.get("qr_flip", False)),
        basis_manual_enabled=bool(s12.get("manual_enabled", False)),
        index_origin=int(cbv.get("index_origin", 0)),
        index_g1=int(cbv.get("index_g1", 3)),
        index_g2=int(cbv.get("index_g2", 4)),
        min_spacing=int(cbv.get("minSpacing", 5)),
        min_absolute_intensity=int(cbv.get("minAbsoluteIntensity", 80)),
        max_num_peaks=int(cbv.get("maxNumPeaks", 60)),
        edge_boundary=int(cbv.get("edgeBoundary", 4)),
        vis_vmin=float(vis.get("vmin", 0.0)),
        vis_vmax=float(vis.get("vmax", 0.995)),
        coordinate_rotation=float(s13.get("coordinate_rotation", 0.0)),
        max_peak_spacing=float(s13.get("max_peak_spacing", 2.0)),
        vrange=list(s13.get("vrange", [-5.0, 5.0])),
        vrange_theta=list(s13.get("vrange_theta", [-5.0, 5.0])),
        strain_layout=str(s13.get("layout", "horizontal")),
        strain_cmap=str(s13.get("cmap", "RdBu_r")),
        strain_cmap_theta=str(s13.get("cmap_theta", "PRGn")),
        strain_show_orientation=bool(s13.get("show_orientation", True)),
        strain_scan_roi_bounds=s13.get("scan_roi_bounds"),
    )


def _overlay_params_dict(params: "CalibrationParams", pd: dict) -> None:
    """Overlay a (partial) params dict onto a CalibrationParams (known fields only)."""
    fields = set(CalibrationParams.__dataclass_fields__)
    for k, v in (pd or {}).items():
        if k in fields:
            try:
                setattr(params, k, v)
            except Exception:
                pass


def scans_from_template(json_path: str) -> list[Scan]:
    """Build a list of Scan objects (with params filled) from a template JSON. Also
    restores the COMPLETE fast4d params (detection params, 6 points, probe, custom
    crystal…) from the per-scan ``params_full`` / ``step_overrides.detection`` blocks
    that ``save_params_template`` writes — these are PRE-origin and were otherwise
    lost (fast_batch only carries the calibration step_overrides)."""
    import json
    cfg = load_template(json_path)
    raw_scans = {}
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            raw_scans = (json.load(fh) or {}).get("scans", {}) or {}
    except Exception:
        raw_scans = {}
    out: list[Scan] = []
    for sc_cfg in cfg.scans:
        sc = Scan(
            name=sc_cfg.name,
            raw_path=sc_cfg.raw_path,
            braggpeaks_path=sc_cfg.braggpeaks_path,
            params=params_from_template_scan(sc_cfg),
        )
        entry = raw_scans.get(sc_cfg.name) if isinstance(raw_scans, dict) else None
        if isinstance(entry, dict):
            h5 = entry.get("h5_path") or ""
            if h5:
                sc.h5_path = str(h5)
        if not sc.h5_path:
            sc.h5_path = calibration_h5_path(sc.raw_path)
        if isinstance(entry, dict):
            # full snapshot wins (round-trips everything); else fall back to the
            # detection block alone.
            if isinstance(entry.get("params_full"), dict):
                _overlay_params_dict(sc.params, entry["params_full"])
            else:
                det = (entry.get("step_overrides") or {}).get("detection")
                if isinstance(det, dict):
                    _overlay_params_dict(sc.params, det)
        out.append(sc)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SESSION JSON — a fast4d-native, round-trippable snapshot of the whole session
# (every scan's paths + full CalibrationParams). Saved next to the results so a
# session can be reopened exactly as left.
# ─────────────────────────────────────────────────────────────────────────────

SESSION_FILENAME = "fast4d_session.json"


def save_session_json(scans: list[Scan], path: str, *, log: Log = None) -> str:
    """Write all scans' paths + calibration params to a session JSON."""
    import json
    from datetime import datetime, timezone
    data = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "scans": [
            {
                "name": sc.name,
                "raw_path": sc.raw_path,
                "h5_path": sc.h5_path,
                "vacuum_path": sc.vacuum_path,
                "braggpeaks_path": sc.braggpeaks_path,
                "results_dir": sc.results_dir,
                "status": sc.status,
                "params": sc.params.to_dict(),
            }
            for sc in scans
        ],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    _log(log, f"Session JSON written: {p}")
    return str(p)


def _scan_lines_seg_map(sc: "Scan") -> dict:
    """scan.lines {lid:[[x0,y0],[x1,y1]]} → {lid:[[x0,y0],[x1,y1]]} (floats), or {}."""
    out = {}
    for lid, seg in (getattr(sc, "lines", None) or {}).items():
        try:
            (x0, y0), (x1, y1) = seg
            out[str(lid)] = [[float(x0), float(y0)], [float(x1), float(y1)]]
        except Exception:
            pass
    return out


def save_params_template(scans: list[Scan], path: str, *, template_index: int = 0,
                         log: Log = None) -> str:
    """Write ONLY the calibration parameters (the parameter table) to a
    ``Parametros_cal.json`` (version 3) — the same format ``scans_from_template`` /
    the loader read back. Inverse of ``params_from_template_scan``; also carries the
    per-scan ROI, lines (segments) and stress constants when present."""
    import json
    scans_d: dict = {}
    lp_per: dict = {}
    for sc in scans:
        p = sc.params
        c11, c12, c44 = p.stress_constants_gpa()
        ovr = {
            "step10": {"sampling": int(p.origin_sampling),
                       "q_range": [int(v) for v in (p.ellipse_q_range or [])],
                       "use_roi": bool(p.ellipse_use_roi)},
            "step11": {"px_guess": float(p.q_px), "kmax": float(p.q_kmax),
                       "kpow": float(p.q_kpow), "use_roi": bool(p.q_use_roi)},
            "step12": {"choose_basis_vectors": {
                           "minSpacing": int(p.min_spacing),
                           "minAbsoluteIntensity": int(p.min_absolute_intensity),
                           "maxNumPeaks": int(p.max_num_peaks),
                           "edgeBoundary": int(p.edge_boundary),
                           "vis_params": {"vmin": float(p.vis_vmin), "vmax": float(p.vis_vmax)},
                           "index_origin": int(p.index_origin),
                           "index_g1": int(p.index_g1),
                           "index_g2": int(p.index_g2)},
                       "qr_rotation": float(p.qr_rotation),
                       "qr_flip": bool(p.qr_flip),
                       "manual_enabled": bool(p.basis_manual_enabled)},
            "step13": {"coordinate_rotation": float(p.coordinate_rotation),
                       "max_peak_spacing": float(p.max_peak_spacing),
                       "vrange": list(p.vrange),
                       "vrange_theta": list(p.vrange_theta),
                       "layout": str(p.strain_layout),
                       "cmap": str(p.strain_cmap),
                       "cmap_theta": str(p.strain_cmap_theta),
                       "show_orientation": bool(p.strain_show_orientation),
                       "scan_roi_bounds": (list(p.strain_scan_roi_bounds)
                                           if p.strain_scan_roi_bounds else None)},
            "stress": {"mode": "plane_stress", "c11_gpa": float(c11),
                       "c12_gpa": float(c12), "c44_gpa": float(c44)},
        }
        seg = _scan_lines_seg_map(sc)
        if seg:
            lp_per[sc.name] = seg
            ovr["lines"] = {"segments": {lid: {"x0": s[0][0], "y0": s[0][1],
                                               "x1": s[1][0], "y1": s[1][1]}
                                         for lid, s in seg.items()},
                            "width": 3, "image_source": "adf"}
        # also store the COMPLETE params (detection params, 6 points, probe source,
        # custom crystal, stress units…) — the step_overrides above only cover the
        # calibration steps, so pre-origin (detection / 6-points) was being lost.
        ovr["detection"] = {
            "probe_source": p.probe_source,
            "six_points": [list(map(float, pt)) for pt in (p.six_points or [])],
            "detect_min_absolute_intensity": int(p.detect_min_absolute_intensity),
            "detect_min_relative_intensity": float(p.detect_min_relative_intensity),
            "detect_min_peak_spacing": int(p.detect_min_peak_spacing),
            "detect_edge_boundary": int(p.detect_edge_boundary),
            "detect_sigma": float(p.detect_sigma),
            "detect_max_num_peaks": int(p.detect_max_num_peaks),
            "detect_subpixel": p.detect_subpixel,
            "detect_corr_power": float(p.detect_corr_power),
            "detect_cuda": bool(p.detect_cuda),
        }
        scans_d[sc.name] = {
            "raw_path": sc.raw_path or "",
            "braggpeaks_path": sc.braggpeaks_path or "",
            "h5_path": sc.h5_path or calibration_h5_path(sc.raw_path) or "",
            "center_guess": list(p.center_guess) if p.center_guess else [128.0, 128.0],
            "roi_bounds": list(p.roi_bounds) if p.roi_bounds else [],
            "step_overrides": ovr,
            "params_full": p.to_dict(),         # complete fast4d snapshot (round-trips ALL fields)
        }
    ti = max(0, min(int(template_index), len(scans) - 1)) if scans else 0
    fixed_lp = lp_per.get(scans[ti].name, {}) if scans else {}
    data = {
        "version": 3,
        "template_index": ti,
        "options": {},
        "fixed_line_profiles": fixed_lp,
        "line_profiles_per_scan": lp_per,
        "scans": scans_d,
        "template_roi_bounds": (list(scans[ti].params.roi_bounds)
                                if scans and scans[ti].params.roi_bounds else []),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)
    _log(log, f"Params template written: {out} ({len(scans_d)} scan(s))")
    return str(out)


def params_from_json(path: str) -> "CalibrationParams | None":
    """Load ONE scan's CalibrationParams from a params/session JSON (for per-file
    calibration in the loader — different samples, individual calibrations)."""
    try:
        scans = load_session_json(path)
    except Exception:
        try:
            scans = scans_from_template(path)
        except Exception:
            scans = []
    return scans[0].params if scans else None


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED METADATA — store the WHOLE analysis (params/calibrations/ROI/lines/drift/
# strain ranges/state) INSIDE the .h5 so the file is self-describing: reopen it and
# everything is restored, and someone else can analyze it WITHOUT the params JSON or
# the raw data. Written as a root dataset 'fast4d_metadata' (a JSON string); py4DSTEM
# ignores the extra dataset.
# ─────────────────────────────────────────────────────────────────────────────

FAST4D_META_KEY = "fast4d_metadata"


def _scan_metadata_dict(scan: "Scan") -> dict:
    """The complete per-scan analysis snapshot (params + lines + ROI + drift + state)."""
    return {
        "fast4d_version": 1,
        "name": scan.name,
        "status": scan.status,
        "params": scan.params.to_dict(),          # all calibrations, ROI, q_px, QR,
                                                  # detection, vrange (strain ranges),
                                                  # stress, custom_crystal, 6 points…
        "lines": {str(lid): [[float(s[0][0]), float(s[0][1])],
                             [float(s[1][0]), float(s[1][1])]]
                  for lid, s in (getattr(scan, "lines", None) or {}).items()},
        "area_roi": list(getattr(scan, "area_roi", None) or []),
        "area_rois": {str(rid): [float(b) for b in bounds]
                      for rid, bounds in scan_area_rois(scan).items()},
        "drift": list(scan.drift) if getattr(scan, "drift", None) else None,
    }


def _meta_h5_targets(scan: "Scan", path: str | None = None) -> list:
    """h5 files to carry the metadata: braggpeaks.h5 (re-calibratable) + the virtual
    h5 (preview). Both, so whichever the user shares is self-describing."""
    if path:
        return [str(path)]
    out: list = []
    for p in (scan.braggpeaks_path, scan.h5_path):
        if (p and Path(p).suffix.lower() in H5_SUFFIXES and Path(p).is_file()
                and str(p) not in out):
            out.append(str(p))
    return out


def embed_metadata_h5(scan: "Scan", path: str | None = None, *, log: Log = None) -> list:
    """Write the full analysis snapshot into the scan's .h5 (root dataset
    ``fast4d_metadata``). Returns the files written."""
    import h5py
    import json
    blob = json.dumps(_scan_metadata_dict(scan), default=str)
    written: list = []
    for p in _meta_h5_targets(scan, path):
        try:
            with h5py.File(p, "a") as f:
                if FAST4D_META_KEY in f:
                    del f[FAST4D_META_KEY]
                f.create_dataset(FAST4D_META_KEY, data=blob)
            written.append(p)
            _log(log, f"[{scan.name}] analysis metadata embedded → {Path(p).name}")
        except Exception as exc:
            _log(log, f"[{scan.name}] embed metadata failed ({Path(p).name}): {exc}")
    if not written:
        _log(log, f"[{scan.name}] no .h5 found to embed metadata into.")
    return written


def read_metadata_h5(path: str) -> dict | None:
    """Read the embedded ``fast4d_metadata`` JSON from an .h5 (or None)."""
    import h5py
    import json
    try:
        with h5py.File(path, "r") as f:
            if FAST4D_META_KEY not in f:
                return None
            raw = f[FAST4D_META_KEY][()]
    except Exception:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        elif isinstance(raw, np.ndarray):
            raw = bytes(raw).decode("utf-8") if raw.dtype.kind in "SV" else str(raw.item())
        return json.loads(raw)
    except Exception:
        return None


def apply_metadata_to_scan(scan: "Scan", meta: dict) -> bool:
    """Overlay an embedded-metadata dict onto a scan (params known-fields + lines +
    area ROI + drift + status)."""
    if not isinstance(meta, dict):
        return False
    pd = meta.get("params")
    if isinstance(pd, dict):
        _overlay_params_dict(scan.params, pd)
    ln = meta.get("lines")
    if isinstance(ln, dict) and ln:
        out: dict = {}
        for k, v in ln.items():
            try:
                out[str(k)] = [[float(v[0][0]), float(v[0][1])],
                               [float(v[1][0]), float(v[1][1])]]
            except Exception:
                pass
        if out:
            scan.lines = out
    if meta.get("area_roi"):
        scan.area_roi = list(meta["area_roi"])
    rois = meta.get("area_rois")
    if isinstance(rois, dict) and rois:
        out_r: dict = {}
        for rid, b in rois.items():
            nb = _normalize_roi(b)
            if nb:
                out_r[str(rid)] = nb
        if out_r:
            scan.area_rois = out_r
    if meta.get("drift"):
        scan.drift = tuple(meta["drift"])
    if meta.get("status"):
        scan.status = meta["status"]
    return True


def load_metadata_for_scan(scan: "Scan", *, log: Log = None) -> bool:
    """Auto-restore a scan's analysis from metadata embedded in its .h5 (if any)."""
    for p in _meta_h5_targets(scan):
        meta = read_metadata_h5(p)
        if meta:
            apply_metadata_to_scan(scan, meta)
            _log(log, f"[{scan.name}] restored embedded analysis from {Path(p).name}")
            return True
    return False


def load_session_json(path: str) -> list[Scan]:
    """Rebuild the Scan list (paths + params) from a session JSON."""
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    fields = set(CalibrationParams.__dataclass_fields__)
    out: list[Scan] = []
    for sd in data.get("scans", []):
        pd = {k: v for k, v in (sd.get("params") or {}).items() if k in fields}
        sc = Scan(
            name=sd.get("name", ""),
            raw_path=sd.get("raw_path", ""),
            h5_path=sd.get("h5_path", ""),
            vacuum_path=sd.get("vacuum_path", ""),
            braggpeaks_path=sd.get("braggpeaks_path", ""),
            params=CalibrationParams(**pd),
        )
        sc.results_dir = sd.get("results_dir", "")
        out.append(sc)
    return out
