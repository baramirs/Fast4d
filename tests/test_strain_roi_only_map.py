"""Strain map of the ROI only (py4DSTEM ``fit_basis_vectors(mask=ROI)``).

With ``strain_roi_only_map`` the with-ROI pass fits g1, g2 only at the ROI scan
positions: the map has data inside the ROI and nothing outside. Inside the ROI the
values are those of the default with-ROI map (the per-pixel fit does not depend on the
mask, and the reference is the same median over the ROI). The without-ROI map is not
touched. Off by default.
"""
import numpy as np

import engine as E
from conftest import MINI_N, MINI_ROI_CENTER


def _roi_only_flow():
    import matplotlib
    matplotlib.use("Agg")
    import synthetic_si110 as S
    bv, truth = S.make_braggvectors(MINI_N)
    sc = S.make_scan(bv, roi_bounds=MINI_ROI_CENTER)
    sc.params.strain_roi_only_map = True
    S.run_flow(sc, log=lambda m: None)
    return sc


def test_default_is_off():
    assert E.CalibrationParams().strain_roi_only_map is False


def test_roi_only_map_has_data_only_inside_the_roi(mini_base):
    sc = _roi_only_flow()
    st = sc.state
    roi = E._reference_roi_mask(sc, st.strainmap_full.rshape)
    raw = np.asarray(st.strain_raw["with_roi"], float)
    valid = raw[4] > 0.5
    assert np.array_equal(valid, roi)                       # nothing outside the ROI
    exx = E.channel_map_2d(sc, "exx", "with_roi")
    assert np.isnan(exx[~roi]).all() and np.isfinite(exx[roi]).all()

    # inside the ROI: identical to the default with-ROI map
    ref = np.asarray(mini_base["scan"].state.strain_raw["with_roi"], float)
    for ch in (0, 1, 2):
        np.testing.assert_allclose(raw[ch][roi], ref[ch][roi], rtol=0, atol=1e-12)

    # the without-ROI map keeps the whole field
    full = np.asarray(st.strain_raw["without_roi"], float)
    assert (full[4] > 0.5).all()
