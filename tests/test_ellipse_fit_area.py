"""F4 of the "Ellipse from area" tool: fit, quality, apply to files, verify on another file.

- ellipse_area.fit_ellipse recovers a/b and the major-axis angle on a recentred ring image
  (both py4DSTEM fits: amorphous halo and 1D ring).
- ellipse_area.residual_ellipse: what is left of a measured ellipse after an applied one.
- engine.fit_ellipse_from_source runs the fit + the existing quality control.
- engine.apply_area_ellipse stores the ellipse (and where it came from) on the chosen scans.
- engine.calibrate_ellipse, with ellipse_source != "bvm", applies the stored ellipse and does
  NOT refit it on the BVM; on the mini synthetic with a detector ellipse the strain comes out
  right only when that ellipse is applied.
- engine.verify_area_ellipse: residual ellipticity of another file after the applied ellipse.
"""
import numpy as np
import pytest

import engine as E
import ellipse_area as EA
import pipeline as P

Q = 128
R_RING, SIG = 30.0, 1.5


def _ring(a_b, theta_deg, c=(63.6, 64.3)):
    qx, qy = np.meshgrid(np.arange(Q, dtype=float), np.arange(Q, dtype=float), indexing="ij")
    dx, dy = qx - c[0], qy - c[1]
    t = np.radians(theta_deg)
    u = (dx * np.cos(t) + dy * np.sin(t)) / a_b
    v = -dx * np.sin(t) + dy * np.cos(t)
    ring = 50.0 * np.exp(-0.5 * ((np.hypot(u, v) - R_RING) / SIG) ** 2)
    return ring + 1000.0 / (1.0 + np.exp((np.hypot(dx, dy) - 3.0) / 0.4))


@pytest.mark.parametrize("method", ["amorphous", "ring"])
def test_fit_recovers_ratio_and_angle(method):
    img = _ring(1.03, 25.0)
    r = EA.fit_ellipse(img, (63.6, 64.3), (R_RING - 6, R_RING + 6), method=method)
    assert r["a_over_b"] == pytest.approx(1.03, abs=0.003)
    assert r["theta_deg"] == pytest.approx(25.0, abs=1.0)
    assert 0.0 <= r["theta_deg"] < 180.0 and r["method"] == method


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        EA.fit_ellipse(_ring(1.03, 25.0), (63.6, 64.3), (24, 36), method="magic")


def test_residual_ellipse():
    p = (0, 0, 1.03, 1.0, np.radians(25.0))
    same = EA.residual_ellipse(p, p)
    assert same["a_over_b"] == pytest.approx(1.0, abs=1e-12)
    none_applied = EA.residual_ellipse(p, None)
    assert none_applied["a_over_b"] == pytest.approx(1.03, abs=1e-9)
    assert none_applied["theta_deg"] == pytest.approx(25.0, abs=1e-6)
    other = EA.residual_ellipse((0, 0, 1.03, 1.0, np.radians(25.0)), (0, 0, 1.01, 1.0, np.radians(25.0)))
    assert other["a_over_b"] == pytest.approx(1.03 / 1.01, abs=1e-9)


def test_fit_from_source_quality():
    src = dict(image=_ring(1.03, 25.0), center=(63.6, 64.3), source="area")
    ok = E.fit_ellipse_from_source(E.Scan(name="s"), src, (R_RING - 6, R_RING + 6))
    assert ok["quality"]["ok"] and ok["a_over_b"] == pytest.approx(1.03, abs=0.003)
    bad = E.fit_ellipse_from_source(E.Scan(name="s"), dict(src, image=_ring(1.12, 25.0)),
                                    (R_RING - 8, R_RING + 8))
    assert not bad["quality"]["ok"]                         # a/b 1.12 > the 1 +- 0.05 limit


def test_apply_area_ellipse_sets_every_target():
    fit = dict(p_ellipse=(63.6, 64.3, 31.0, 30.0, np.radians(25.0)), method="amorphous",
               r_range=(24, 36), a_over_b=31 / 30, theta_deg=25.0)
    src = dict(source="area", recentered=True, origin_source="origin_fit")
    a, b = E.Scan(name="A"), E.Scan(name="B")
    E.apply_area_ellipse([a, b], fit, src, areas=[{"kind": "rect", "bounds": [0, 5, 0, 9]}],
                         source_scan=a)
    for sc in (a, b):
        p = sc.params
        assert p.ellipse_enabled and p.ellipse_source == "area"
        assert p.ellipse_p_fitted == pytest.approx([31.0, 30.0, np.radians(25.0)])
        assert p.ellipse_fit_method == "amorphous" and p.ellipse_recenter is True
        assert p.ellipse_source_file == "A" and p.ellipse_source_rois[0]["kind"] == "rect"


def test_default_source_is_bvm():
    assert E.CalibrationParams().ellipse_source == "bvm"


def _mini(ellipse, source, p_fitted, jitter_px=0.03):
    import synthetic_si110 as S
    bv, truth = S.make_braggvectors(16, ellipse=ellipse, jitter_px=jitter_px)
    sc = S.make_scan(bv, roi_bounds=[4, 11, 4, 11])
    p = sc.params
    p.ellipse_enabled = p_fitted is not None
    p.ellipse_source = source
    p.ellipse_p_fitted = p_fitted
    return sc, truth


def test_calibration_applies_stored_area_ellipse_without_bvm_refit(monkeypatch):
    def _no_refit(*a, **k):
        raise AssertionError("the BVM ellipse fit must not run for an area ellipse")
    monkeypatch.setattr(P, "fit_ellipse_step", _no_refit)
    stored = [1.03, 1.0, float(np.radians(25.0))]
    sc, _ = _mini((1.03, 1.0, 25.0), "area", stored)
    logs = []
    E.run_calibration_sequence(sc, make_figures=False, log=logs.append)
    a, b, th = sc.state.braggpeaks.calibration.get_ellipse()
    assert (a, b, th) == pytest.approx(tuple(stored))
    assert any("area" in m and "ellipse" in m for m in logs)


def test_area_ellipse_corrects_the_strain():
    # exact peak positions (no jitter): with the ellipse applied the strain is the truth
    # (with the 0.03 px jitter both this and the no-ellipse case sit at ~0.15 % rms)
    import synthetic_si110 as S
    ell = (1.03, 1.0, 25.0)
    errs = {}
    for name, pf in (("off", None), ("on", [1.03, 1.0, float(np.radians(25.0))])):
        sc, truth = _mini(ell, "area", pf, jitter_px=0.0)
        S.run_flow(sc, with_roi=True, log=lambda m: None)
        roi = E._reference_roi_mask(sc, sc.state.strainmap_full.rshape)
        raw = np.asarray(sc.state.strain_raw["with_roi"], float)
        v = raw[4] > 0.5
        d = (raw[0] - S.expected_eyy(truth, roi))[v] * 100
        errs[name] = float(np.sqrt(np.mean(d ** 2)))
    assert errs["on"] < 1e-3 and errs["off"] > 1.0


def test_verify_on_another_file(monkeypatch):
    applied = (0, 0, 1.03, 1.0, np.radians(25.0))
    same = dict(image=_ring(1.03, 25.0), center=(63.6, 64.3), source="area")
    other = dict(image=_ring(1.06, 25.0), center=(63.6, 64.3), source="area")
    for src, ok in ((same, True), (other, False)):
        monkeypatch.setattr(E, "ellipse_source_image", lambda *a, _s=src, **k: _s)
        r = E.verify_area_ellipse(E.Scan(name="other"), [{"kind": "rect", "bounds": [0, 3, 0, 3]}],
                                  applied, (R_RING - 6, R_RING + 6))
        assert r["ok"] is ok
    assert r["residual_a_over_b"] == pytest.approx(1.06 / 1.03, abs=0.004)
