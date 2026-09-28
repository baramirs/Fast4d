"""Inter-scan drift from ADF/BF: recovered shifts, sign convention, CSV round trip."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as nd_shift

import drift_estimate as DE


def _field(seed=0, n=256):
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.normal(size=(n, n)), 3.0)
    yy, xx = np.mgrid[0:n, 0:n]
    img += 0.004 * xx                       # illumination gradient (must not bias the shift)
    img[80:180, 60:70] += 2.0               # a wall-like feature
    return img


def _scan(name, adf, bf):
    st = SimpleNamespace(virtual_images={"annular_dark_field": adf, "bright_field": bf},
                         strain_raw={})
    return SimpleNamespace(name=name, adf_cache=adf, state=st, results_dir=None,
                           h5_path=None, braggpeaks_path=None, raw_path=None)


def _series(moves, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    base = _field()
    out = []
    for i, (my, mx) in enumerate(moves):
        adf = nd_shift(base, (my, mx), order=3, mode="nearest") + noise * rng.normal(size=base.shape)
        bf = -0.7 * nd_shift(base, (my, mx), order=3, mode="nearest") + noise * rng.normal(size=base.shape)
        out.append(_scan(f"S{i + 1}", adf, bf))
    return out


MOVES = [(0.0, 0.0), (3.4, -2.1), (6.8, -4.3)]     # feature moves +dy rows / +dx cols


@pytest.mark.parametrize("key", ["adf", "bf", "adf+bf"])
@pytest.mark.parametrize("mode", ["reference", "chained"])
def test_shifts_and_sign(key, mode):
    scans = _series(MOVES)
    res = DE.estimate_drift(scans, scans[0], image_key=key, mode=mode, upsample_factor=20)
    for r, (my, mx) in zip(res, MOVES):
        # registration shift = minus the feature displacement
        assert r["dy"] == pytest.approx(-my, abs=0.15)
        assert r["dx"] == pytest.approx(-mx, abs=0.15)
        if not r["is_reference"]:
            assert r["ncc_after"] > r["ncc_before"]
            assert r["ncc_after"] > 0.8


def test_line_lands_on_feature():
    """engine negates the stored shift: a line at the feature in S1 must move
    with the feature in S3."""
    scans = _series(MOVES)
    res = DE.estimate_drift(scans, scans[0], image_key="adf+bf")
    r3 = res[2]
    line_dx, line_dy = -r3["dx"], -r3["dy"]            # engine._line_drift_shift
    assert line_dy == pytest.approx(MOVES[2][0], abs=0.15)
    assert line_dx == pytest.approx(MOVES[2][1], abs=0.15)


def test_csv_columns(tmp_path):
    scans = _series(MOVES)
    res = DE.estimate_drift(scans, scans[0], image_key="adf")
    p = DE.save_drift_csv(res, tmp_path / "d.csv")
    head = p.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert head[:3] == ["stem", "shift_dy_px", "shift_dx_px"]
    assert not {"name", "file", "filename", "scan", "scan_id"} & set(head)


def test_summary():
    scans = _series(MOVES)
    res = DE.estimate_drift(scans, scans[0], image_key="adf")
    s = DE.summary(res, px_nm=1.148)
    assert s["max_shift_px"] == pytest.approx(np.hypot(6.8, 4.3), abs=0.2)
    assert s["mean_step_nm"] == pytest.approx(np.hypot(3.4, 2.1) * 1.148, abs=0.3)


def test_overlays_sharpen_after_correction(tmp_path):
    """After the correction the scans coincide: channels agree (grey), before
    they do not."""
    scans = _series(MOVES, noise=0.0)
    res = DE.estimate_drift(scans, scans[0], image_key="adf+bf")
    before, after, ch, names = DE.overlay_maps(scans, res, "adf+bf")
    assert ch == "adf" and names == ["S1", "S2", "S3"]
    inner = (slice(20, -20), slice(20, -20))
    spread = lambda rgb: float(np.mean(rgb[inner].max(-1) - rgb[inner].min(-1)))
    assert spread(after) < 0.3 * spread(before)
    out = DE.save_overlays(before, after, names, tmp_path, results=res, px_nm=1.148)
    assert [p.name for p in out] == ["drift_overlay_before.png", "drift_overlay_after.png"]
    assert all(p.stat().st_size > 1000 for p in out)
