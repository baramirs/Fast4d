"""G3: analysis lines and ROIs must average only pixels with a lattice fit.

Sanitizing leaves pixels without a fit (vacuum, failed fit) at 0.0 in strain_raw,
with channel 4 = 0. line_profiles / roi_profiles read the tensor without that mask,
so every invalid pixel entered the average as 0 % strain (Scan01: line mean exx
-0.145 % instead of -0.213 %; synthetic ROI R_menos -1.07 % instead of -6.94 %).
Mini synthetic: vacuum band on columns 0-5; the line and ROI cross its edge.
"""
import numpy as np

import engine as E


def _valid_and_maps(flow, label="without_roi"):
    # float32 like fast_artifacts._as_hw3, which every profile reads through
    raw = np.asarray(flow["scan"].state.strain_raw[label]).astype(np.float32).astype(float)
    return raw[4] > 0.5, {"eyy": raw[0] * 100, "exx": raw[1] * 100, "exy": raw[2] * 100}


def test_roi_half_in_vacuum_gives_the_mean_of_the_valid_pixels(mini_vacuum):
    sc = mini_vacuum["scan"]
    bounds = [0, 11, 4, 19]                      # columns 0-11: half vacuum
    sc.area_rois = {"R_edge": bounds}
    try:
        stats = E.roi_profiles(sc, "without_roi")["R_edge"]
    finally:
        sc.area_rois = {}
    valid, maps = _valid_and_maps(mini_vacuum)
    m = E.roi_mask(valid.shape, bounds)
    for ch in ("eyy", "exx", "exy"):
        assert stats[ch]["n"] == int((m & valid).sum())
        assert np.isclose(stats[ch]["mean"], float(np.mean(maps[ch][m & valid])), rtol=0, atol=1e-9)


def test_line_crossing_into_vacuum_skips_invalid_samples(mini_vacuum):
    sc = mini_vacuum["scan"]
    sc.lines = {"L1": [[0.0, 12.0], [23.0, 12.0]]}   # along row 12, columns 0 -> 23
    sc.line_widths = {"L1": 1}
    try:
        pr = E.line_profiles(sc, "without_roi")["L1"]
    finally:
        sc.lines, sc.line_widths = {}, {}
    valid, maps = _valid_and_maps(mini_vacuum)
    eyy = np.asarray(pr["eyy"], float)
    assert np.all(np.isnan(eyy[:5]))                 # samples on vacuum pixels 0-4
    assert np.all(np.isfinite(eyy[6:]))
    np.testing.assert_allclose(eyy[6:], maps["eyy"][12, 6:], rtol=0, atol=1e-9)
    assert int(pr["n_valid"]) == int(np.isfinite(eyy).sum())
    assert np.isclose(np.nanmean(eyy[6:]), float(np.mean(maps["eyy"][12, 6:])), atol=1e-9)


def test_batch_line_profiles_skip_invalid_samples(mini_vacuum):
    import fast_batch as FB
    raw = mini_vacuum["scan"].state.strain_raw["without_roi"]
    res = FB._extract_line_profiles({"without_roi": raw}, None,
                                    {"L1": [[0.0, 12.0], [23.0, 12.0]]}, line_width=1)
    eyy = np.asarray(res["without_roi"]["L1"]["eyy"], float)
    valid, maps = _valid_and_maps(mini_vacuum)
    assert np.all(np.isnan(eyy[:5])) and np.all(np.isfinite(eyy[6:]))
    np.testing.assert_allclose(eyy[6:] * 100, maps["eyy"][12, 6:], rtol=0, atol=1e-6)


def test_channel_maps_are_nan_where_the_fit_failed(mini_vacuum):
    vac = mini_vacuum["truth"]["vacuum"]
    for ch in ("eyy", "exx", "exy"):
        m = E.channel_map_2d(mini_vacuum["scan"], ch, "without_roi")
        assert np.all(np.isnan(m[vac])) and np.all(np.isfinite(m[~vac]))


def test_profiles_unchanged_where_everything_is_valid(mini_base):
    sc = mini_base["scan"]
    sc.lines = {"L1": [[2.0, 3.0], [20.0, 19.0]]}
    sc.line_widths = {"L1": 3}
    try:
        pr = E.line_profiles(sc, "without_roi")["L1"]
    finally:
        sc.lines, sc.line_widths = {}, {}
    raw = np.asarray(mini_base["scan"].state.strain_raw["without_roi"]).astype(np.float32)
    dist, ref = E._line_samples(raw[0], [[2.0, 3.0], [20.0, 19.0]], width=3)   # old formula
    np.testing.assert_allclose(np.asarray(pr["eyy"]), ref * 100.0, rtol=1e-12, atol=0)
