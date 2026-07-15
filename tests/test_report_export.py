"""Tests for per-channel report export (no PIL collage crops)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeParams:
    strain_cmap = "RdBu_r"
    strain_cmap_theta = "PRGn"
    vrange = [-5.0, 5.0]
    vrange_theta = [-5.0, 5.0]
    stress_vmax = 0.0
    stress_units = "GPa"


class _FakeState:
    def __init__(self, hw3, stress=None):
        self.strain_raw = {"without_roi": hw3, "with_roi": hw3}
        self.stress_tensors_pa = stress or {}


class _FakeScan:
    def __init__(self, name, hw3, *, figures=None):
        self.name = name
        self.params = _FakeParams()
        self.state = _FakeState(hw3)
        self.figures = figures or {}
        self.figure_spill = {}

    def ensure_state(self):
        return self.state


def _make_fig(title: str = "t"):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_title(title)
    return fig


@pytest.fixture
def fake_scans():
    rng = np.random.default_rng(0)
    hw3 = rng.normal(0, 0.01, size=(32, 32, 3)).astype(np.float32)
    return [_FakeScan("ScanA", hw3), _FakeScan("ScanB", hw3)]


@pytest.fixture
def fake_scans_with_figs(fake_scans):
    sc = fake_scans[0]
    sc.figures = {
        "origin": _make_fig("origin"),
        "report_line_L1_eyy_without_roi": _make_fig("line"),
    }
    return fake_scans


def test_prepare_assets_and_pdf(tmp_path, fake_scans):
    from tools.report_export.assets import prepare_map_assets, iter_channel_images
    from tools.report_export.build import build_report

    assets = tmp_path / "export_assets"
    man = prepare_map_assets(fake_scans, assets, dpi=80)
    rows = iter_channel_images(man)
    assert len(rows) >= 6  # 2 scans × 3 strain channels × at least without_roi
    assert all(r["path"].is_file() for r in rows)
    assert not any("maps_split" in str(r["path"]) for r in rows)

    pdf = build_report(scans=fake_scans, assets_dir=assets,
                       out_path=tmp_path / "out.pdf", fmt="pdf", dpi=80)
    assert pdf.is_file() and pdf.stat().st_size > 1000


def test_pptx_from_assets(tmp_path, fake_scans):
    from tools.report_export.build import build_report

    out = build_report(scans=fake_scans, assets_dir=tmp_path / "ea",
                       out_path=tmp_path / "out.pptx", fmt="pptx", dpi=72)
    assert out.is_file() and out.suffix == ".pptx"


def test_docx_from_assets(tmp_path, fake_scans):
    pytest.importorskip("docx")
    from tools.report_export.build import build_report

    out = build_report(scans=fake_scans, assets_dir=tmp_path / "ea_docx",
                       out_path=tmp_path / "out.docx", fmt="docx", dpi=72)
    assert out.is_file() and out.suffix == ".docx"
    assert out.stat().st_size > 500


def test_include_flags_calib_and_reports(tmp_path, fake_scans_with_figs):
    from tools.report_export.assets import prepare_export_assets, iter_report_images

    assets = tmp_path / "ea_flags"
    man = prepare_export_assets(
        fake_scans_with_figs, assets, dpi=72,
        include_maps=False, include_calib=True, include_reports=True)
    rows = iter_report_images(man)
    kinds = {r["kind"] for r in rows}
    assert kinds == {"calib", "report"}
    assert all(r["path"].is_file() for r in rows)
    assert any(r["channel"] == "origin" for r in rows)
    assert any("report_line" in str(r["channel"]) for r in rows)


def test_maps_only_skips_figs(tmp_path, fake_scans_with_figs):
    from tools.report_export.assets import prepare_export_assets, iter_report_images

    man = prepare_export_assets(
        fake_scans_with_figs, tmp_path / "ea_maps", dpi=72,
        include_maps=True, include_calib=False, include_reports=False)
    rows = iter_report_images(man)
    assert rows
    assert all(r["kind"] in ("strain", "stress") for r in rows)


def test_shared_vrange_across_panels(tmp_path, fake_scans):
    """Export panels must use GUI vrange, not per-array percentiles."""
    from tools.report_export.assets import prepare_export_assets, load_manifest

    for sc in fake_scans:
        sc.params.vrange = [-1.25, 1.25]
        sc.params.vrange_theta = [-0.5, 0.5]

    man = prepare_export_assets(
        fake_scans, tmp_path / "ea_vr", dpi=72,
        include_maps=True, include_calib=False, include_reports=False)
    # All strain ε channels share the same clim
    for sc_name, sc in man["scans"].items():
        for label, lab in sc["labels"].items():
            for ch in ("exx", "eyy", "exy"):
                meta = lab["channels"][ch]
                assert meta["vmin"] == pytest.approx(-1.25)
                assert meta["vmax"] == pytest.approx(1.25)
