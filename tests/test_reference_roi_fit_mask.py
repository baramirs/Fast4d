"""G1: the with-ROI reference must use ROI AND the lattice-fit mask, as py4DSTEM does.

py4DSTEM ``get_strain(gvects=ROI)`` (the notebook) takes the median of g1, g2 over
``ROI AND StrainMap.mask``. Fast4D captured the median over the ROI alone, so pixels
without a fit (g = 0: vacuum, amorphous) pulled the reference and shifted the WHOLE
map (Scan01: -1.07 % eyy with 25 % failed pixels in the ROI, +100 % above 50 %).
Mini synthetic: vacuum band on columns 0-5, calibration ROI with 4 of 10 columns in it.
"""
import numpy as np
import pytest

import engine as E
import pipeline as P


def _notebook_reference(sm, roi):
    from py4DSTEM.process.strain.latticevectors import get_reference_g1g2
    g1, g2 = get_reference_g1g2(sm.g1g2_map, np.logical_and(roi, sm.mask))
    return np.asarray(g1, float), np.asarray(g2, float)


def _notebook_strain(sm, roi, coordinate_rotation):
    """What the notebook's get_strain(gvects=ROI) gives, without touching ``sm``."""
    from py4DSTEM.process.strain.latticevectors import (
        get_rotated_strain_map, get_strain_from_reference_g1g2)
    g1, g2 = _notebook_reference(sm, roi)
    s = get_strain_from_reference_g1g2(sm.g1g2_map, g1, g2)
    th = np.radians(coordinate_rotation)
    r = get_rotated_strain_map(s, xaxis_x=np.cos(th), xaxis_y=np.sin(th), flip_theta=False)
    return {k: r.get_slice(k).data for k in ("e_xx", "e_yy", "e_xy")}


def test_fixture_roi_really_crosses_the_vacuum_edge(mini_vacuum):
    sm = mini_vacuum["scan"].state.strainmap_full
    roi = E._reference_roi_mask(mini_vacuum["scan"], sm.rshape)
    failed = roi & ~np.asarray(sm.mask, bool)
    assert 0.3 < failed.sum() / roi.sum() < 0.5
    assert np.array_equal(~np.asarray(sm.mask, bool), mini_vacuum["truth"]["vacuum"])


def test_captured_reference_equals_py4dstem_roi_and_fit_mask(mini_vacuum):
    sm = mini_vacuum["scan"].state.strainmap_full
    roi = E._reference_roi_mask(mini_vacuum["scan"], sm.rshape)
    g1, g2 = P.capture_reference_g12_from_strainmap(sm, roi)
    n1, n2 = _notebook_reference(sm, roi)
    assert np.array_equal(g1, n1) and np.array_equal(g2, n2)


def test_with_roi_map_equals_notebook_get_strain_with_roi(mini_vacuum):
    st = mini_vacuum["scan"].state
    sm = st.strainmap_full
    roi = E._reference_roi_mask(mini_vacuum["scan"], sm.rshape)
    raw = np.asarray(st.strain_raw["with_roi"], float)
    valid = raw[4] > 0.5
    cr = float(st.crystal_frame_cards["with_roi"]["x_angle_applied_deg"])
    nb = _notebook_strain(sm, roi, cr)
    assert valid.sum() == (~mini_vacuum["truth"]["vacuum"]).sum()
    np.testing.assert_allclose(raw[1][valid], nb["e_xx"][valid], rtol=0, atol=1e-12)
    np.testing.assert_allclose(raw[0][valid], nb["e_yy"][valid], rtol=0, atol=1e-12)
    np.testing.assert_allclose(raw[2][valid], nb["e_xy"][valid], rtol=0, atol=1e-12)


def test_with_roi_eyy_matches_the_truth(mini_vacuum):
    import synthetic_si110 as S
    st = mini_vacuum["scan"].state
    roi = E._reference_roi_mask(mini_vacuum["scan"], st.strainmap_full.rshape)
    raw = np.asarray(st.strain_raw["with_roi"], float)
    valid = raw[4] > 0.5
    d = (raw[0] - S.expected_eyy(mini_vacuum["truth"], roi))[valid] * 100
    # % : no offset of the whole map (was +1.58 % before the fix; 0.0002 % without the
    # 0.03 px peak jitter, ~0.1 % with it: noise of the median over a 48 px reference)
    assert abs(float(np.mean(d))) < 0.25


def test_excluded_pixels_are_reported(mini_vacuum):
    sm = mini_vacuum["scan"].state.strainmap_full
    roi = E._reference_roi_mask(mini_vacuum["scan"], sm.rshape)
    logs = []
    P.capture_reference_g12_from_strainmap(sm, roi, log=logs.append)
    n_roi, n_bad = int(roi.sum()), int((roi & ~np.asarray(sm.mask, bool)).sum())
    msg = " ".join(logs)
    assert "WARNING" in msg and f"{n_bad} of {n_roi}" in msg


def test_roi_without_enough_fitted_pixels_is_refused(mini_vacuum):
    sm = mini_vacuum["scan"].state.strainmap_full
    roi = np.zeros(sm.rshape, bool)
    roi[:, :6] = True                                         # vacuum only
    with pytest.raises(RuntimeError, match="lattice fit"):
        P.capture_reference_g12_from_strainmap(sm, roi)
    roi = np.zeros(sm.rshape, bool)
    roi[0:4, 4:10] = True                                     # 24 px, only 16 fitted
    with pytest.raises(RuntimeError, match="lattice fit"):
        P.capture_reference_g12_from_strainmap(sm, roi)
