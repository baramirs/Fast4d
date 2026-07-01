from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkflowState:
    """Persistent in-memory state for the notebook-equivalent workflow."""

    raw_mib_path: Path | None = None
    precomputed_h5_path: Path | None = None
    vacuum_mib_path: Path | None = None

    datacube: Any = None
    visualcube: Any = None
    vacuumcube: Any = None
    probe: Any = None

    probe_alpha: float | None = None
    probe_qx0: float | None = None
    probe_qy0: float | None = None

    # Probe built from a user-defined vacuum region on a virtual image (ADF/BF) of the main datacube.
    # When set, Step 2 can call `datacube.get_vacuum_probe(ROI=mask)` instead of loading a separate vacuum .mib.
    probe_source: str | None = None  # "vacuum_mib" | "bf_roi" | "synthetic" | "mean_dp_patch"
    probe_bf_roi_bounds: tuple[int, int, int, int] | None = None
    probe_bf_roi_mask: Any = None

    virtual_images: dict[str, Any] = field(default_factory=dict)

    braggpeaks_path: Path | None = None
    braggpeaks: Any = None
    use_existing_braggpeaks: bool = False

    image_pixel_size: float | None = None
    image_pixel_units: str = "px"

    roi_bounds: tuple[int, int, int, int] | None = None
    roi_mask: Any = None
    bragg_points: list[tuple[float, float]] = field(default_factory=list)
    bragg_rxs: tuple[int, ...] = ()
    bragg_rys: tuple[int, ...] = ()
    selected_disks: Any = None

    origin_sampling: int = 1
    center_guess: tuple[int, int] | None = None
    origin_measurement: Any = None
    origin_fit: tuple[Any, Any, Any, Any] | None = None
    bvm_raw: Any = None
    bvm_centered: Any = None

    ellipse_sampling: int = 1
    ellipse_q_range: tuple[int, int] = (40, 51)
    ellipse_use_roi: bool = True
    ellipse_bvm: Any = None
    p_ellipse: Any = None

    q_pixel_size: float | None = None
    q_pixel_units: str = "A^-1"
    # Set when loading e.g. ``simulate_4d_cube.py`` output + sidecar ``*_meta.json`` (see pipeline.load_data_step).
    recommended_q_pixel_A_inv_per_px: float | None = None
    q_crystal: str = "Si"
    strain_basis_params: dict[str, Any] = field(default_factory=lambda: {
        "choose_basis_vectors": {
            "minSpacing": 5,
            "minAbsoluteIntensity": 80,
            "maxNumPeaks": 60,
            "edgeBoundary": 4,
            "vis_params": {"vmin": 0.0, "vmax": 0.995},
        },
        "qr_rotation": 0.0,
        "qr_flip": False,
        "manual_enabled": False,
    })
    strain_params: dict[str, Any] = field(default_factory=lambda: {
        "set_max_peak_spacing": {"max_peak_spacing": 2},
        "fit_basis_vectors": {},
        "get_strain": {
            "coordinate_rotation": 90,
            "layout": "horizontal",
            "vrange": [-2, 2],
            "vrange_theta": [-45.0, 45.0],
            "cmap": "RdBu_r",
            "cmap_theta": "PRGn",
        },
        "show_orientation": True,  # Fast4D-only flag; not forwarded to py4DSTEM get_strain()
    })
    strainmap_full: Any = None
    strainmap_roi: Any = None
    strain_figures: dict[str, Any] = field(default_factory=dict)
    strain_arrays: dict[str, Any] = field(default_factory=dict)
    strain_raw: dict[str, Any] = field(default_factory=dict)  # label -> raw strainmap_g1g2.data (often 3D/structured)
    line_profiles_px: dict[str, Any] = field(default_factory=dict)  # label -> {1:(p0,p1), 2:(p0,p1)}
    fixed_line_profiles_px: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None
    # Step 14 reference ROI on εxx map: pixel box x0,x1,y0,y1 with x1/y1 exclusive (numpy slice semantics).
    strain_roi_rect: tuple[int, int, int, int] | None = None
    strain_roi_mask: Any = None
    # Optional sector on the scan: mask braggpeaks outside this region before StrainMap (notebook: mask_in_R(mask=~ROI)).
    strain_scan_roi_bounds: tuple[int, int, int, int] | None = None
    strain_scan_roi_mask: Any = None
    # Optional: (g1, g2) reciprocal lattice vectors from another scan / ROI (e.g. naked Si), for
    # ``StrainMap.get_strain(gvects=(g1,g2))`` on the current Bragg peaks. Each inner tuple is (qx, qy).
    strain_external_reference_g12: tuple[tuple[float, float], tuple[float, float]] | None = None
    strain_use_external_reference: bool = False

    # Optional Step 15: Cauchy stress σ_ij (Pa) from ε via cubic Hooke's law (θ not used).
    stress_tensors_pa: dict[str, Any] = field(default_factory=dict)
    stress_meta: dict[str, Any] = field(default_factory=dict)
    stress_figures: dict[str, Any] = field(default_factory=dict)
    stress_line_profiles_px: dict[str, Any] = field(default_factory=dict)
    stress_line_profile_data: dict[str, Any] = field(default_factory=dict)  # key -> {distance_px, sigma_xx, sigma_yy, unit}

    figures: dict[str, Any] = field(default_factory=dict)

    # Last Step 12 choose_basis_vectors figure(s); used for reliable PNG export (pyplot figures).
    basis_preview_figures: list[Any] = field(default_factory=list)

    detect_params: dict[str, Any] = field(default_factory=lambda: {
        "minAbsoluteIntensity": 3,
        "minRelativeIntensity": 0.05,
        "minPeakSpacing": 10,
        "edgeBoundary": 10,
        "sigma": 0.0,
        "maxNumPeaks": 50,
        "subpixel": "poly",
        "corrPower": 1.0,
        "CUDA": True,
        "CUDA_batched": True,
    })

    # Step 7 preview only: same semantics as notebook `view_filter` + cmap (not passed to find_Bragg_disks).
    disk_preview_params: dict[str, Any] = field(default_factory=lambda: {
        "view_mode": "highpass",
        "p_lo": 1.0,
        "p_hi": 99.8,
        "gamma": 0.45,
        "hp_sigma": 6.0,
        "cmap": "inferno",
    })

    # Optional GPU acceleration for Step 9 BVM fallback (only used if py4DSTEM braggpeaks lacks .histogram()).
    bvm_params: dict[str, Any] = field(default_factory=lambda: {
        "use_cupy_histogram": False,
    })

    def has_datacube(self) -> bool:
        return self.datacube is not None

    def has_visualcube(self) -> bool:
        return self.visualcube is not None

    def has_probe(self) -> bool:
        return self.probe is not None

    def reset_data_products(self) -> None:
        self.datacube = None
        self.visualcube = None
        self.virtual_images.clear()
        self.recommended_q_pixel_A_inv_per_px = None

    def reset_probe_products(self) -> None:
        self.vacuumcube = None
        self.probe = None
        self.probe_alpha = None
        self.probe_qx0 = None
        self.probe_qy0 = None
        self.probe_source = None
        self.probe_bf_roi_bounds = None
        self.probe_bf_roi_mask = None

    def reset_bragg_products(self) -> None:
        self.braggpeaks = None
        self.selected_disks = None
        self.bragg_points.clear()
        self.bragg_rxs = ()
        self.bragg_rys = ()
        self.reset_origin_products()

    def reset_origin_products(self) -> None:
        self.center_guess = None
        self.origin_measurement = None
        self.origin_fit = None
        self.bvm_raw = None
        self.bvm_centered = None
        self.reset_ellipse_products()

    def reset_ellipse_products(self) -> None:
        self.ellipse_bvm = None
        self.p_ellipse = None
        self.reset_strain_products()

    def reset_strain_products(self) -> None:
        self.strainmap_full = None
        self.strainmap_roi = None
        self.strain_figures.clear()
        self.strain_arrays.clear()
        self.strain_raw.clear()
        self.line_profiles_px.clear()
        self.fixed_line_profiles_px = None
        self.strain_roi_mask = None
        self.strain_scan_roi_bounds = None
        self.strain_scan_roi_mask = None
        self.strain_external_reference_g12 = None
        self.strain_use_external_reference = False
        self.stress_tensors_pa.clear()
        self.stress_meta.clear()
        self.stress_figures.clear()
        self.stress_line_profiles_px.clear()
        self.stress_line_profile_data.clear()
        self.basis_preview_figures.clear()
        self.figures.clear()
