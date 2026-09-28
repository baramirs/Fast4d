"""Reproducibility experiment: drift correction must remove the drift part of
the apparent irreproducibility, and leave only the noise."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as nd_shift

import drift_estimate as DE
import reproducibility as R

MOVES = [(0.0, 0.0), (2.6, -3.3), (5.1, -6.2)]      # feature displacement (rows, cols)
NOISE = 0.01                                        # % strain noise per pixel


def _truth(n=256):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    exx = 0.8 / (1 + np.exp(-(xx - 128) / 3.0)) - 0.4          # interface step, %
    eyy = 0.2 * np.sin(yy / 17.0) + 0.1 * np.cos(xx / 23.0)
    exy = 0.05 * np.sin((xx + yy) / 31.0)
    rng = np.random.default_rng(0)
    img = gaussian_filter(rng.normal(size=(n, n)), 3) + 1.5 * (xx > 128)
    return exx, eyy, exy, img


def _scans():
    exx, eyy, exy, img = _truth()
    rng = np.random.default_rng(1)
    out = []
    for k, (my, mx) in enumerate(MOVES):
        mv = lambda a: nd_shift(a, (my, mx), order=3, mode="nearest")
        S = np.dstack([mv(eyy) + NOISE * rng.normal(size=eyy.shape),
                       mv(exx) + NOISE * rng.normal(size=eyy.shape),
                       mv(exy) + NOISE * rng.normal(size=eyy.shape)]) / 100.0
        S[:, :6] = 0.0                                          # a failed-fit stripe
        adf = mv(img) + 0.05 * rng.normal(size=img.shape)
        bf = -0.8 * mv(img) + 0.05 * rng.normal(size=img.shape)
        st = SimpleNamespace(strain_raw={"without_roi": S},
                             virtual_images={"annular_dark_field": adf, "bright_field": bf})
        out.append(SimpleNamespace(name=f"S{k + 1}", state=st, adf_cache=adf, drift=None,
                                   params=SimpleNamespace(r_px_pm=1148.0), results_dir=None,
                                   raw_path=None, h5_path=None, braggpeaks_path=None,
                                   lines={}, area_rois={}, area_roi=None))
    out[0].lines = {"L1": [[40, 120], [220, 120]]}
    out[0].area_rois = {"R1": [30, 90, 60, 200], "R2": [170, 230, 60, 200]}
    return out


def _with_drift(scans):
    res = DE.estimate_drift(scans, scans[0], image_key="adf+bf")
    for s, r in zip(scans, res):
        s.drift = (r["dx"], r["dy"])
    return scans


def test_drift_correction_removes_drift_part():
    scans = _with_drift(_scans())
    res = R.run(scans, scans[0])
    L = res.lines["L1"]["exx"]
    # without correction the step moves along the line: large spread at the step
    assert np.nanmax(L["no_drift"]["std"]) > 5 * np.nanmax(L["drift"]["std"])
    # with correction only the noise is left
    assert L["drift"]["s_mean"] < 0.01
    for key in ("rmse_vs_ref",):
        nd = max(p[key] for p in L["no_drift"]["per_scan"])
        dr = max(p[key] for p in L["drift"]["per_scan"])
        assert dr < 0.5 * nd
    m = res.maps["eyy"]
    assert m["drift"]["s_mean"] < 0.5 * m["no_drift"]["s_mean"]
    rm = [p["rmse"] for p in m["drift"]["per_scan"] if p["scan"] != "S1"]
    assert max(rm) < 0.03


def test_roi_and_contrast():
    scans = _with_drift(_scans())
    res = R.run(scans, scans[0])
    c = res.contrast["R1-R2"]["exx"]["drift"]["summary"]
    assert c["mean"] == pytest.approx(-0.8, abs=0.02)            # left - right of the step
    assert c["std"] < 0.01


def test_invalid_positions_excluded_and_quality():
    scans = _with_drift(_scans())
    res = R.run(scans, scans[0])
    q = res.quality[0]["valid_fraction"]
    assert q == pytest.approx(1 - 6 / 256, abs=1e-6)


def test_tables_and_export(tmp_path):
    scans = _with_drift(_scans())
    res = R.run(scans, scans[0])
    df = R.summary_table(res)
    assert {"object", "channel", "metric", "no_drift", "drift", "unit"} <= set(df.columns)
    assert len(R.drift_table(res)) == 3
    out = R.export(res, tmp_path)
    names = {p.name for p in out}
    assert "reproducibility_summary.csv" in names and "reproducibility_line_L1.png" in names


def test_no_drift_loaded_gives_identical_columns():
    scans = _scans()
    res = R.run(scans, scans[0])
    L = res.lines["L1"]["exx"]
    assert L["no_drift"]["s_mean"] == pytest.approx(L["drift"]["s_mean"])
    assert res.notes
