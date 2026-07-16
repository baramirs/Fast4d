"""Tests for plugins.indexing (peaks upsample, registry, no cross-imports)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_three_plugins():
    from plugins.indexing.registry import get_plugin, list_plugins

    plugs = list_plugins()
    ids = [p.id for p in plugs]
    assert ids == ["index_bvm_unknown", "index_bvm_known", "orient_peaks"]
    assert get_plugin("index_bvm_unknown").label
    assert get_plugin("orient_peaks").id == "orient_peaks"
    with pytest.raises(KeyError):
        get_plugin("nope")


def test_orientation_peaks_does_not_import_bvm_indexing():
    """Orient motor must not depend on Index BVM (shared peaks live in plugins)."""
    src = (ROOT / "orientation_peaks.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    assert "bvm_indexing" not in imported


def test_find_peaks_image_upsample_recovers_offset_peak():
    pytest.importorskip("py4DSTEM")
    pytest.importorskip("scipy")
    from plugins.indexing.peaks import find_peaks

    # py4DSTEM get_maxima_2D: x = axis0 (row), y = axis1 (col)
    true_x, true_y = 50.5, 40.5
    yy, xx = np.mgrid[0:96, 0:96]  # yy=row, xx=col
    bvm = 200.0 * np.exp(
        -((yy - true_x) ** 2 + (xx - true_y) ** 2) / (2 * 1.2 ** 2)
    )

    m1 = find_peaks(
        bvm,
        min_spacing=5,
        min_absolute_intensity=20,
        max_num_peaks=5,
        edge_boundary=4,
        subpixel="pixel",
        upsample_factor=1,
        image_upsample=1,
    )
    m2 = find_peaks(
        bvm,
        min_spacing=5,
        min_absolute_intensity=20,
        max_num_peaks=5,
        edge_boundary=4,
        subpixel="pixel",
        upsample_factor=1,
        image_upsample=4,
    )
    assert len(m1) >= 1 and len(m2) >= 1
    err1 = float(np.hypot(m1["x"][0] - true_x, m1["y"][0] - true_y))
    err2 = float(np.hypot(m2["x"][0] - true_x, m2["y"][0] - true_y))
    # Upsampled detection should be at least as close (usually closer) to true center
    assert err2 <= err1 + 0.15
    assert err2 < 0.6


def test_bvm_indexing_find_bvm_maxima_delegates():
    pytest.importorskip("py4DSTEM")
    import bvm_indexing as bix

    bvm = np.zeros((64, 64), dtype=float)
    # row=30, col=40 → py4DSTEM x=30, y=40
    bvm[30, 40] = 500.0
    m = bix.find_bvm_maxima(
        bvm,
        min_spacing=5,
        min_absolute_intensity=10,
        max_num_peaks=3,
        edge_boundary=2,
        subpixel="pixel",
        upsample_factor=1,
        image_upsample=1,
    )
    assert len(m) >= 1
    assert abs(float(m["x"][0]) - 30.0) < 1.5
    assert abs(float(m["y"][0]) - 40.0) < 1.5


def test_apply_proposal_writes_params():
    from types import SimpleNamespace

    from plugins.indexing.apply import apply_proposal_to_scan
    from plugins.indexing.types import BasisProposal

    params = SimpleNamespace(
        index_origin=0,
        index_g1=0,
        index_g2=0,
        basis_manual_enabled=False,
        qr_rotation=0.0,
        coordinate_rotation=0.0,
    )
    scan = SimpleNamespace(name="t", params=params, indexing_result=None)
    prop = BasisProposal(
        plugin_id="index_bvm_unknown",
        index_origin=1,
        index_g1=2,
        index_g2=3,
        g1_px=np.array([1.0, 0.0]),
        g2_px=np.array([0.0, 1.0]),
        suggested_qr_rotation_deg=12.5,
        suggested_coordinate_rotation_deg=12.5,
        raw_result=SimpleNamespace(ok=True),
    )
    apply_proposal_to_scan(scan, prop)
    assert params.index_origin == 1
    assert params.index_g1 == 2
    assert params.index_g2 == 3
    assert params.basis_manual_enabled is True
    assert abs(params.qr_rotation - 12.5) < 1e-9
    assert scan.indexing_result is not None
