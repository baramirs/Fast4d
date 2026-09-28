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


def test_q_pixel_far_from_the_seed_warns():
    w = E.q_pixel_fit_warnings(px_guess=0.0224, px_fit=0.030512, use_roi=True, n_patterns=162)
    assert any("seed" in s for s in w) and any("ROI" in s for s in w)
    assert E.q_pixel_fit_warnings(px_guess=0.0224, px_fit=0.021572, use_roi=False,
                                  n_patterns=65536) == []
