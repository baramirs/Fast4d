"""Shared types for indexing plugins (BVM in → Basis proposal out)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BvmContext:
    """Calibrated Bragg Vector Map + crystal hints for an indexer plugin."""

    bvm: np.ndarray
    origin_px: np.ndarray  # (qx, qy) absolute BVM px
    Q_pixel: float
    Q_units: str = "A^-1"
    accel_voltage: float = 300_000.0
    lattice_a: float | None = None
    crystal: Any = None  # py4DSTEM Crystal or None
    crystal_name: str = ""
    crystal_key: str = ""
    # Peak detection (shared)
    min_spacing: float = 20.0
    min_absolute_intensity: float = 80.0
    max_num_peaks: int = 60
    edge_boundary: float = 40.0
    image_upsample: int = 1  # 1|2|4 spatial sampling before maxima
    # Orientation / lattice knobs (plugins read what they need)
    zone_axis: list[int] | None = None
    real_axis_h: list[int] | None = None
    real_axis_v: list[int] | None = None
    proj_x_lattice: list[int] | None = None
    qr_rotation_deg: float = 0.0
    qr_flip: bool = False
    tol_px: float = 2.0
    seed: int = 0
    k_max: float = 1.2
    angle_step_zone_axis: float = 4.0
    angle_step_in_plane: float = 4.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class BasisProposal:
    """Common Send contract: propose origin/g1/g2 for choose_basis_vectors."""

    plugin_id: str
    index_origin: int
    index_g1: int
    index_g2: int
    g1_px: np.ndarray
    g2_px: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)
    suggested_qr_rotation_deg: float | None = None
    suggested_coordinate_rotation_deg: float | None = None
    raw_result: Any = None  # IndexingResult | OrientationPeaksResult | …
    figure_key: str = "indexing"
