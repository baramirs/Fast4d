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


def test_channel_display_levels_honours_gui_vrange_not_data_percentiles():
    """Live maps must use the GUI range even when the array has larger outliers."""
    import numpy as np
    import engine as E

    class P:
        vrange = [-1.25, 1.25]
        vrange_theta = [-10.0, 10.0]
        stress_vmax = 2.5
        strain_cmap = "RdBu_r"
        strain_cmap_theta = "PRGn"

    class S:
        params = P()

    arr = np.array([[-8.0, 8.0], [0.0, 0.4]], dtype=float)
    assert E.channel_display_levels(S(), "eyy", arr) == (-1.25, 1.25)
    assert E.channel_display_levels(S(), "exx", arr) == (-1.25, 1.25)
    assert E.channel_display_levels(S(), "theta", arr) == (-10.0, 10.0)
    assert E.channel_display_levels(S(), "sxx", arr) == (-2.5, 2.5)
    assert E.channel_display_cmap(S(), "eyy") == "RdBu_r"
    assert E.channel_display_cmap(S(), "orientation") == "PRGn"
    assert E.channel_display_cmap(S(), "adf") == "gray"


def test_channel_display_levels_adf_and_stress_auto():
    import numpy as np
    import engine as E

    class P:
        vrange = [-2.0, 2.0]
        vrange_theta = [-5.0, 5.0]
        stress_vmax = 0.0  # auto

    class S:
        params = P()

    adf = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=float)
    lo, hi = E.channel_display_levels(S(), "adf", adf)
    assert lo == pytest.approx(float(np.nanpercentile(adf, 1)))
    assert hi == pytest.approx(float(np.nanpercentile(adf, 99)))

    stress = np.array([[-4.0, 0.0], [0.0, 4.0]], dtype=float)
    slo, shi = E.channel_display_levels(S(), "syy", stress)
    pad = float(np.nanpercentile(np.abs(stress), 98))
    assert slo == pytest.approx(-pad)
    assert shi == pytest.approx(pad)


def test_single_line_map_figure_clim_matches_gui_vrange():
    """Send-to-Report / live-line map must not auto-percentile past the GUI vrange."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import engine as E

    hw3 = np.zeros((8, 8, 3), dtype=float)
    hw3[..., 0] = 0.08  # ε_yy = 8% after ×100 — outside ±1.25

    class P:
        vrange = [-1.25, 1.25]
        vrange_theta = [-5.0, 5.0]
        stress_vmax = 0.0
        strain_cmap = "RdBu_r"
        strain_cmap_theta = "PRGn"
        stress_units = "GPa"

    class St:
        strain_raw = {"without_roi": hw3}
        stress_tensors_pa = {}

    class S:
        name = "s"
        params = P()
        state = St()
        lines = {"L1": [[1.0, 2.0], [6.0, 2.0]]}
        adf_cache = None

        def ensure_state(self):
            return self.state

    fig = E.build_single_line_map_figure(S(), "L1", "eyy", "without_roi")
    assert fig.axes[0].images, "expected an imshow on the map axis"
    assert fig.axes[0].images[0].get_clim() == (-1.25, 1.25)


def test_r_pixel_scale_uses_emd_pm():
    """Velox PixelSize is pm/px; axis distances must become nm."""
    import numpy as np
    import engine as E

    class P:
        r_px_pm = 143.5

    class S:
        params = P()
        state = None

    scale, unit = E.r_pixel_scale(S())
    assert unit == "nm"
    assert scale == pytest.approx(0.1435)
    dist, u = E.scale_line_distance([10.0, 20.0], S())
    assert u == "nm"
    assert np.asarray(dist)[0] == pytest.approx(1.435)


def test_r_pixel_scale_zero_stays_pixels():
    import engine as E

    class P:
        r_px_pm = 0.0

    class S:
        params = P()
        state = None

    scale, unit = E.r_pixel_scale(S())
    assert (scale, unit) == (1.0, "px")


def test_apply_r_pixel_from_params_zero_does_not_propagate():
    """r_px_pm=0 must not push state.image_pixel_size back onto the scan."""
    import engine as E

    class P:
        r_px_pm = 0.0

    class S:
        params = P()
        state = None
        name = "t"

        def ensure_state(self):
            raise AssertionError("must not propagate when r_px_pm is 0")

    E.apply_r_pixel_from_params(S())


def test_param_spec_exposes_r_px_pm():
    from param_spec import PARAM_SPEC

    fields = [s.field for s in PARAM_SPEC["tools"]]
    assert "r_px_pm" in fields
