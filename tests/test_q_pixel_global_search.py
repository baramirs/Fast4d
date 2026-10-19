"""Q-pixel global search (benchmark findings 1, 2, 4, 5; docs/BENCHMARK_FIX_PLAN.md).

py4DSTEM calibrate_pixel_size is a local fit: from the guess 0.022 (true 0.020) it lands
on 0.0343 (+71 %), and on an unstrained crystal it stays at 0.020938 (+4.7 %). The global
search must find the true value in both, flag them, and leave good cases unchanged."""
import numpy as np
import pytest

import engine as E
import synthetic_si110 as S
import driver as D

TRUE_PX = S.Q_PIX
# py4DSTEM's local fit itself scatters ~0.1-0.15 % with the start point (same basin):
# that, not the search, sets the precision; strain does not depend on it.
Q_TOL = 2e-3


def _scan(e_max=0.02, q_guess=0.021, seed=0):
    bv, truth = S.make_braggvectors(24, e_max=e_max, jitter_px=0.03, seed=seed)
    sc = S.make_scan(bv, q_guess=q_guess)
    return sc, truth


def _calibrated(**kw):
    sc, truth = _scan(**kw)
    E.run_calibration_sequence(sc, make_figures=False, log=lambda m: None)
    return sc, truth


def test_unstrained_crystal_no_longer_stuck_at_the_guess():
    sc, _ = _calibrated(e_max=0.0)
    assert abs(sc.params.q_px_fitted / TRUE_PX - 1) < Q_TOL      # was +4.7 %
    assert sc.params.q_px_check["status"] == "differs"          # the old fit is reported


def test_guess_10pct_off_finds_the_true_ring_family():
    sc, _ = _calibrated(q_guess=0.022, seed=0)
    assert abs(sc.params.q_px_fitted / TRUE_PX - 1) < Q_TOL      # was +71 %
    chk = sc.params.q_px_check
    assert chk["status"] == "differs"
    assert abs(chk["px_from_guess"] / TRUE_PX - 1) > 0.2        # the local fit alone fails


def test_good_case_is_ok_and_unchanged():
    sc, _ = _calibrated()
    assert abs(sc.params.q_px_fitted / TRUE_PX - 1) < Q_TOL
    assert sc.params.q_px_check["status"] == "ok"


def test_test_mode_does_not_change_the_applied_px():
    sc, _ = _calibrated()
    pl = E._pipeline()
    before = pl._get_q_pixel_size(sc.state.braggpeaks.calibration)
    res = E.q_pixel_global_search(sc, px_guess=0.025, span_rel=0.3, tol_rel=0.01,
                                  k_max=sc.params.q_kmax, kpow=sc.params.q_kpow,
                                  use_roi=False, apply=False, log=lambda m: None)
    assert abs(res["best"]["px_fit"] / TRUE_PX - 1) < Q_TOL
    assert pl._get_q_pixel_size(sc.state.braggpeaks.calibration) == before
    fig = E.q_pixel_search_figure(res)
    assert fig.axes and len(res["grid"]) == len(res["match"])


def test_search_span_is_validated():
    sc, _ = _calibrated()
    with pytest.raises(ValueError, match="span"):
        E.q_pixel_global_search(sc, px_guess=0.02, span_rel=0.0, tol_rel=0.01, k_max=1.6,
                                kpow=1.0, use_roi=False, apply=False)


def test_with_roi_strain_without_global_first():
    """Finding 5: with-ROI right after calibration crashed (basis StrainMap, no g1g2_map)."""
    sc, _ = _scan()
    sc.params.roi_bounds = [8, 16, 8, 16]
    E.run_calibration_sequence(sc, make_figures=False, log=lambda m: None)
    E.compute_strain(sc, use_roi=True, log=lambda m: None)       # used to raise
    direct = np.asarray(sc.state.strain_raw["with_roi"], float).copy()
    E.compute_strain(sc, use_roi=False, log=lambda m: None)
    E.compute_strain(sc, use_roi=True, log=lambda m: None)
    again = np.asarray(sc.state.strain_raw["with_roi"], float)
    np.testing.assert_array_equal(direct, again)


@pytest.mark.parametrize("answer, ok", [(False, False), (True, True)])
def test_driver_asks_before_strain_when_flagged(tmp_path, answer, ok):
    import py4DSTEM
    bv, _ = S.make_braggvectors(24, e_max=0.02, jitter_px=0.03, seed=0)
    path = tmp_path / "bp.h5"
    py4DSTEM.save(str(path), bv, mode="o")
    sc = S.make_scan(None, q_guess=0.022)
    sc.braggpeaks_path = str(path)
    asked = []

    def confirm(scan, msg):
        asked.append(msg)
        return answer

    opts = D.ComputeOptions(do_with_roi=False, make_figures=False, figure_mode="off",
                            save=False, save_figures=False, save_braggpeaks=False)
    out = D.compute_scan(sc, opts, log=lambda m: None, confirm=confirm)
    assert len(asked) == 1 and "Q pixel" in asked[0]
    assert out.ok is ok
    if not ok:
        assert out.error == "cancelled"
