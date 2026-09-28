"""M1-M3: ellipse calibration (centre, quality control, 'apply' mode).

M1  ellipse_center [0, 0] (as stored in Scan01.h5) was taken as a given centre, so
    fit_ellipse_1D returned its seed (0, 0, 50, 50, 90 deg) and that was applied.
M2  the fitted ellipse was applied without any check (Scan01, 162 px ROI: a/b = 1.55,
    centre 12 px off -> eyy RMS 0.41 %); the Q-pixel fit had no warning either.
M3  'apply' mode ignored the ellipse: 'fit' and 'apply' gave different maps.
Mini synthetic with a known ellipse: a/b = 1.02 with a at 30 deg from +qx.
"""
import copy

import numpy as np
import pytest

import engine as E
import pipeline as P
import synthetic_si110 as S

ELL = (1.02, 1.0, 30.0)                      # generator: stretch 2 % along 30 deg
N = 20


def _p_true():
    qx0, qy0 = S.ORIGIN0
    return (qx0, qy0, 51.0, 50.0, float(np.radians(ELL[2])))


def _scan(ellipse=ELL):
    bv, truth = S.make_braggvectors(N, ellipse=ellipse)
    sc = S.make_scan(bv, roi_bounds=[4, 16, 4, 16])
    return sc, truth


@pytest.fixture(scope="module")
def origin_scan():
    sc, _ = _scan()
    E.set_roi(sc, log=lambda m: None)
    E.calibrate_origin(sc, make_figure=False, log=lambda m: None)
    return sc


# ── M1 ─────────────────────────────────────────────────────────────────────────

def _captured_center(monkeypatch, st, center):
    import py4DSTEM
    seen = {}

    def fake(bvm, center=None, fitradii=None, mask=None):
        seen["center"] = tuple(float(v) for v in center)
        seen["origin"] = tuple(float(v) for v in bvm.origin)
        return (center[0], center[1], 50.0, 50.0, 0.0)

    monkeypatch.setattr(py4DSTEM.process.calibration, "fit_ellipse_1D", fake)
    P.fit_ellipse_step(st, (40, 60), sampling=1, use_roi=False, center=center)
    return seen


@pytest.mark.parametrize("center", [(0.0, 0.0), [0, 0], (-3194.0, 2681.0), (float("nan"), 3.0)])
def test_missing_or_bogus_center_falls_back_to_bvm_origin(monkeypatch, origin_scan, center):
    st = copy.copy(origin_scan.state)
    seen = _captured_center(monkeypatch, st, center)
    assert seen["center"] == seen["origin"]


def test_a_real_center_is_kept(monkeypatch, origin_scan):
    st = copy.copy(origin_scan.state)
    probe = _captured_center(monkeypatch, st, None)
    c = (probe["origin"][0] + 1.5, probe["origin"][1] - 0.5)
    assert _captured_center(monkeypatch, st, c)["center"] == c


# ── M2 ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("p, ok", [
    ((128.0, 128.0, 51.0, 50.0, 0.5), True),        # a/b 1.02, on the origin
    ((128.0, 128.0, 77.5, 50.0, 0.5), False),       # a/b 1.55 (Scan01, 162 px ROI)
    ((140.0, 128.0, 51.0, 50.0, 0.5), False),       # centre 12 px off
    ((128.0, 128.0, 500.0, 490.0, 0.5), False),     # scale far from the ring
])
def test_ellipse_quality_control(p, ok):
    q = P.ellipse_fit_quality(p, origin=(128.0, 128.0), q_range=(40, 60))
    assert q["ok"] is ok
    assert bool(q["reasons"]) is (not ok)


def test_rejected_ellipse_is_not_applied(monkeypatch):
    import py4DSTEM
    sc, _ = _scan()
    sc.params.ellipse_enabled = True
    sc.params.ellipse_use_roi = False
    E.set_roi(sc, log=lambda m: None)
    E.calibrate_origin(sc, make_figure=False, log=lambda m: None)
    monkeypatch.setattr(py4DSTEM.process.calibration, "fit_ellipse_1D",
                        lambda bvm, center=None, fitradii=None, mask=None:
                        (center[0], center[1], 77.5, 50.0, 0.3))
    with pytest.raises(RuntimeError, match="rejected"):
        E.calibrate_ellipse(sc, make_figure=False, log=lambda m: None)
    bp = sc.state.braggpeaks
    assert not bp.calstate.get("ellipse", False)
    assert sc.params.ellipse_p_fitted is None


def test_q_pixel_far_from_the_seed_warns():
    w = E.q_pixel_fit_warnings(px_guess=0.0224, px_fit=0.030512, use_roi=True, n_patterns=162)
    assert any("seed" in s for s in w) and any("ROI" in s for s in w)
    assert E.q_pixel_fit_warnings(px_guess=0.0224, px_fit=0.021572, use_roi=False,
                                  n_patterns=65536) == []


# ── M3 ─────────────────────────────────────────────────────────────────────────

def _fit_with_known_ellipse(sc):
    """'fit' path, with the known ellipse applied via the ellipse tool (a single-crystal
    BVM cannot constrain an ellipse fit: its rings have only two spot directions)."""
    E.set_roi(sc, log=lambda m: None)
    E.calibrate_origin(sc, make_figure=False, log=lambda m: None)
    E.apply_ellipse_fit(sc, _p_true(), r_range=(40, 60), make_figure=False)
    E.calibrate_q_pixel(sc, refit=True, make_figure=False, log=lambda m: None)
    E.calibrate_basis(sc, make_figure=False, log=lambda m: None)
    E.compute_strain(sc, use_roi=False, log=lambda m: None)
    return np.asarray(sc.state.strain_raw["without_roi"], float)


@pytest.fixture(scope="module")
def fit_and_apply():
    sc_fit, truth = _scan()
    raw_fit = _fit_with_known_ellipse(sc_fit)
    sc_apply, _ = _scan()
    sc_apply.params = copy.deepcopy(sc_fit.params)
    logs = []
    E.apply_calibration(sc_apply, log=logs.append)
    E.compute_strain(sc_apply, use_roi=False, log=logs.append)
    raw_apply = np.asarray(sc_apply.state.strain_raw["without_roi"], float)
    return sc_fit, raw_fit, raw_apply, truth


def test_fit_stores_the_ellipse_it_applied(fit_and_apply):
    sc_fit = fit_and_apply[0]
    a, b, th = sc_fit.params.ellipse_p_fitted
    assert (a, b) == (51.0, 50.0) and np.isclose(th, np.radians(30.0))


def test_apply_mode_gives_the_same_map_as_fit(fit_and_apply):
    _, raw_fit, raw_apply, _ = fit_and_apply
    v = raw_fit[4] > 0.5
    assert np.array_equal(v, raw_apply[4] > 0.5)
    np.testing.assert_allclose(raw_apply[:3][:, v], raw_fit[:3][:, v], rtol=0, atol=1e-9)


def test_the_known_ellipse_recovers_the_truth(fit_and_apply):
    _, raw_fit, _, truth = fit_and_apply
    v = raw_fit[4] > 0.5
    d_eyy = (raw_fit[0] - S.expected_eyy(truth))[v] * 100
    assert np.sqrt(np.mean(d_eyy ** 2)) < 0.2 and np.sqrt(np.mean((raw_fit[1][v] * 100) ** 2)) < 0.2


def test_apply_with_ellipse_enabled_but_no_fit_is_refused():
    sc, _ = _scan()
    sc.params.ellipse_enabled = True
    sc.params.ellipse_p_fitted = None
    with pytest.raises(RuntimeError, match="ellipse"):
        E.apply_calibration(sc, log=lambda m: None)
