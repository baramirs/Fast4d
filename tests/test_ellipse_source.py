"""F3 of the "Ellipse from area" tool: the image the ellipse is fitted on.

ellipse_area.mean_dp averages the patterns of a scan mask, each one shifted to a common beam
centre first (descan would otherwise smear the ring); ellipse_area.beam_centers_com measures
that centre when no origin map exists. engine.ellipse_source_image wraps both for a Scan:
"area" (raw 4D data + user areas), "dp_mean" / "dp_max" (precomputed, not re-centrable).
Mini synthetic: 10 x 10 scan, 96 x 96 patterns, direct beam + elliptical ring, known descan.
"""
import numpy as np
import pytest

import engine as E
import ellipse_area as EA

N, Q = 10, 96
R_RING, SIG = 22.0, 1.2
A_B, THETA = 1.04, np.radians(30.0)


def _origins():
    u = np.linspace(-1, 1, N)
    qx0 = 47.6 + 2.0 * u[:, None] + 0 * u[None, :]
    qy0 = 48.3 + 0 * u[:, None] - 2.0 * u[None, :]
    return qx0, qy0


def _pattern(cx, cy):
    qx, qy = np.meshgrid(np.arange(Q, dtype=float), np.arange(Q, dtype=float), indexing="ij")
    dx, dy = qx - cx, qy - cy
    c, s = np.cos(THETA), np.sin(THETA)
    u = (dx * c + dy * s) / A_B
    v = -dx * s + dy * c
    ring = 50.0 * np.exp(-0.5 * ((np.hypot(u, v) - R_RING) / SIG) ** 2)
    beam = 1000.0 / (1.0 + np.exp((np.hypot(dx, dy) - 3.0) / 0.4))
    return ring + beam


@pytest.fixture(scope="module")
def cube():
    qx0, qy0 = _origins()
    data = np.zeros((N, N, Q, Q), np.float32)
    for i in range(N):
        for j in range(N):
            data[i, j] = _pattern(qx0[i, j], qy0[i, j])
    return data, qx0, qy0


def _left_half():
    m = np.zeros((N, N), bool)
    m[:, :5] = True
    return m


def test_mean_without_recentering_is_the_plain_mean(cube):
    data, *_ = cube
    m = _left_half()
    r = EA.mean_dp(data, m, origins=None, recenter=False)
    np.testing.assert_allclose(r["image"], data[m].astype(np.float64).mean(axis=0), rtol=0, atol=1e-6)
    assert r["n_patterns"] == m.sum() and r["recentered"] is False


def test_recentering_restores_the_single_pattern_ring(cube):
    data, qx0, qy0 = cube
    m = _left_half()
    rec = EA.mean_dp(data, m, origins=(qx0, qy0), recenter=True)
    cx, cy = rec["center"]
    assert cx == pytest.approx(qx0[m].mean()) and cy == pytest.approx(qy0[m].mean())
    ideal = _pattern(cx, cy)
    raw = data[m].mean(axis=0)
    err_rec = np.sqrt(np.mean((rec["image"] - ideal) ** 2))
    err_raw = np.sqrt(np.mean((raw - ideal) ** 2))
    assert err_rec < 0.25 * err_raw                      # descan smearing removed


def test_beam_centres_from_center_of_mass(cube):
    data, qx0, qy0 = cube
    m = _left_half()
    cqx, cqy = EA.beam_centers_com(data, m, guess=(48.0, 48.0), radius=8)
    assert np.nanmax(np.abs(cqx[m] - qx0[m])) < 0.1
    assert np.nanmax(np.abs(cqy[m] - qy0[m])) < 0.1
    assert np.isnan(cqx[~m]).all()


def test_empty_mask_raises(cube):
    data, *_ = cube
    with pytest.raises(ValueError):
        EA.mean_dp(data, np.zeros((N, N), bool))


class _FakeCube:
    def __init__(self, data):
        self.data = data
        self.Rshape = data.shape[:2]


def _scan_with_raw(monkeypatch, cube, *, origin_fit=True):
    data, qx0, qy0 = cube
    sc = E.Scan(name="mini", raw_path="mini.mib")
    st = sc.ensure_state()
    if origin_fit:
        st.origin_fit = (qx0, qy0, np.zeros_like(qx0), np.zeros_like(qy0))
    sc.params.center_guess = [48.0, 48.0]
    monkeypatch.setattr(E._pipeline(), "_load_raw_datacube", lambda *a, **k: _FakeCube(data))
    return sc


def test_engine_area_uses_origin_fit(monkeypatch, cube):
    sc = _scan_with_raw(monkeypatch, cube)
    areas = [{"kind": "rect", "bounds": [0, 4, 0, 9]}]           # columns 0-4 = left half
    r = E.ellipse_source_image(sc, "area", areas=areas)
    assert r["source"] == "area" and r["recentered"] is True
    assert r["origin_source"] == "origin_fit" and r["n_patterns"] == 50
    data, qx0, qy0 = cube
    direct = EA.mean_dp(data, _left_half(), origins=(qx0, qy0), recenter=True)
    np.testing.assert_allclose(r["image"], direct["image"], rtol=0, atol=1e-6)


def test_engine_area_without_origin_fit_uses_com(monkeypatch, cube):
    sc = _scan_with_raw(monkeypatch, cube, origin_fit=False)
    r = E.ellipse_source_image(sc, "area", areas=[{"kind": "rect", "bounds": [0, 4, 0, 9]}])
    assert r["origin_source"] == "beam_com"
    data, qx0, qy0 = cube
    m = _left_half()
    assert r["center"][0] == pytest.approx(qx0[m].mean(), abs=0.05)
    assert r["center"][1] == pytest.approx(qy0[m].mean(), abs=0.05)


def test_engine_area_needs_areas(monkeypatch, cube):
    sc = _scan_with_raw(monkeypatch, cube)
    with pytest.raises(ValueError):
        E.ellipse_source_image(sc, "area", areas=[])


def test_engine_dp_mean_and_max_are_not_recentered(cube):
    data, *_ = cube
    sc = E.Scan(name="mini")
    st = sc.ensure_state()
    st.virtual_images = {"dp_mean": data.mean(axis=(0, 1)), "dp_max": data.max(axis=(0, 1))}
    for src in ("dp_mean", "dp_max"):
        r = E.ellipse_source_image(sc, src)
        np.testing.assert_array_equal(r["image"], st.virtual_images[src])
        assert r["source"] == src and r["recentered"] is False and r["center"] is not None


def test_engine_unknown_source_raises(cube):
    with pytest.raises(ValueError):
        E.ellipse_source_image(E.Scan(name="x"), "bvm_mean")
