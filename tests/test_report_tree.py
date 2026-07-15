"""Tests for Report tree taxonomy and on-demand Send-to-Report keys."""
from __future__ import annotations

import pytest

from qt_report import (
    classify_figure_key,
    _is_report_key,
    _label_for_key,
    _STRAIN_CHANNELS,
    _STRESS_CHANNELS,
)


@pytest.mark.parametrize(
    "key,branch",
    [
        ("origin", "calib"),
        ("indexing", "calib"),
        ("strain_without_roi", "map_without"),
        ("stress_with_roi", "map_with"),
        ("report_line_L1_eyy_without_roi", "reports"),
        ("report_roi_group_R1_eyy_with_roi", "reports"),
        ("line_profiles", "legacy"),
        ("maps_with_lines", "legacy"),
        ("line_group_L1", "reports"),
    ],
)
def test_classify_figure_key(key, branch):
    assert classify_figure_key(key) == branch


def test_report_key_helpers():
    assert _is_report_key("report_line_L1_eyy_without_roi")
    assert not _is_report_key("strain_without_roi")
    assert "Line map" in _label_for_key("report_line_L1_eyy_without_roi")


def test_list_figure_keys_no_resolve(monkeypatch):
    import engine as E

    class FakeScan:
        name = "s"
        figures = {"origin": object(), "strain_without_roi": object()}
        figure_spill = {"basis": "/tmp/x.png"}

    keys = E.list_figure_keys(FakeScan())
    assert keys[0] == "origin"  # FIGURE_ORDER first
    assert "basis" in keys
    assert "strain_without_roi" in keys


def test_map_channel_leaves_cover_strain_and_stress():
    strain_ids = [c for c, _ in _STRAIN_CHANNELS]
    stress_ids = [c for c, _ in _STRESS_CHANNELS]
    assert strain_ids == ["exx", "eyy", "exy", "orientation"]
    assert stress_ids == ["sxx", "syy", "sxy"]
    # Filter needle "exx" must match a channel id
    assert any("exx" in c for c in strain_ids)


def test_roi_ref_labels():
    import engine as E

    assert "Theoretical" in E.roi_ref_label("without_roi")
    assert "Experimental" in E.roi_ref_label("with_roi")
    assert E.ROI_REF_LABELS["without_roi"] != E.ROI_REF_LABELS["with_roi"]


def test_channel_clim_uses_gui_vrange():
    import engine as E

    class P:
        vrange = [-2.0, 2.0]
        vrange_theta = [-1.5, 1.5]
        stress_vmax = 3.0

    class S:
        params = P()

    assert E.channel_clim(S(), "exx") == (-2.0, 2.0)
    assert E.channel_clim(S(), "orientation") == (-1.5, 1.5)
    assert E.channel_clim(S(), "sxx") == (-3.0, 3.0)
