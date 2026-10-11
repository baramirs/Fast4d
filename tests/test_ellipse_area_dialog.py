"""F5 of the "Ellipse from area" tool: the window (offscreen) and its Qt-free helpers.

- ellipse_area.radial_profile: azimuthal average, optionally in the ellipse-corrected frame;
  the corrected ring is narrower. suggest_ring finds the ring outside the direct beam.
- qt_ellipse_area.EllipseAreaDialog: areas drawn as pyqtgraph ROIs come back as
  engine.area_mask dicts (rect, ellipse, polygon, exclude); the DP sources disable
  re-centring; Build -> Fit -> Apply / Verify call the engine with the drawn areas.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

import engine as E
import ellipse_area as EA

Q = 128
R_RING = 30.0


def _ring(a_b=1.03, theta_deg=25.0, c=(63.6, 64.3), sig=1.2):
    qx, qy = np.meshgrid(np.arange(Q, dtype=float), np.arange(Q, dtype=float), indexing="ij")
    dx, dy = qx - c[0], qy - c[1]
    t = np.radians(theta_deg)
    u = (dx * np.cos(t) + dy * np.sin(t)) / a_b
    v = -dx * np.sin(t) + dy * np.cos(t)
    ring = 50.0 * np.exp(-0.5 * ((np.hypot(u, v) - R_RING) / sig) ** 2)
    return ring + 1000.0 / (1.0 + np.exp((np.hypot(dx, dy) - 3.0) / 0.4))


# ── Qt-free helpers ───────────────────────────────────────────────────────────
def test_radial_profile_corrected_ring_is_narrower():
    # a/b 1.06 on a 30 px ring spreads the circular average over ~1.8 px; a 0.8 px ring
    # (FWHM 1.9 px) shows it clearly (with a/b 1.03 and FWHM 2.8 px it is only ~3 % wider)
    img = _ring(a_b=1.06, sig=0.8)
    p = EA.fit_ellipse(img, (63.6, 64.3), (24, 36), method="amorphous")["p_ellipse"]
    r, i0 = EA.radial_profile(img, (63.6, 64.3))
    r1, i1 = EA.radial_profile(img, (63.6, 64.3), p_ellipse=p)
    w0 = EA.peak_fwhm(r, i0, (24, 36))
    w1 = EA.peak_fwhm(r1, i1, (24, 36))
    ideal = 2.3548 * 0.8                                     # FWHM of the undistorted ring
    assert abs(w1 - ideal) < 0.25 and w0 - w1 > 0.5         # measured: 2.00 vs 2.63 px


def test_suggest_ring_skips_the_direct_beam():
    r, i = EA.radial_profile(_ring(), (63.6, 64.3))
    r0, r1 = EA.suggest_ring(r, i, r_min=8.0, half_width=6.0)
    assert r0 < R_RING < r1 and r1 - r0 == pytest.approx(12.0)


# ── the window ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Console:
    def __init__(self):
        self.lines = []

    def log(self, m, *_a, **_k):
        self.lines.append(str(m))


class _Host(QtWidgets.QWidget):
    def __init__(self, scans):
        super().__init__()
        self._scans = scans
        self._active = 0
        self._busy = False
        self._console = _Console()

    def active_scan(self):
        return self._scans[self._active]

    def _run_async(self, work, *, label, on_done=None):
        try:
            r = work()
        except Exception as exc:          # same contract as Fast4DWindow._run_async
            r = exc
        if on_done is not None:
            on_done(r)


@pytest.fixture
def dlg(app, monkeypatch):
    import qt_ellipse_area as QE
    adf = np.random.default_rng(0).normal(100, 5, (32, 32))
    monkeypatch.setattr(E, "cached_adf", lambda sc: adf)
    scans = [E.Scan(name="A"), E.Scan(name="B"), E.Scan(name="C")]
    d = QE.EllipseAreaDialog(_Host(scans))
    yield d
    d.close()


def test_drawn_areas_round_trip(dlg):
    dlg.add_area("rect", [4.0, 14.0, 6.0, 20.0, 30.0])
    dlg.add_area("ellipse", {"center": [20.0, 10.0], "radii": [5.0, 3.0], "angle": 15.0})
    dlg.add_area("polygon", [[1.0, 1.0], [8.0, 2.0], [5.0, 9.0]])
    a = dlg.areas()
    assert [x["kind"] for x in a] == ["rect", "ellipse", "polygon"]
    assert a[0]["bounds"] == pytest.approx([4.0, 14.0, 6.0, 20.0]) and a[0]["angle"] == pytest.approx(30.0)
    assert a[1]["center"] == pytest.approx([20.0, 10.0]) and a[1]["radii"] == pytest.approx([5.0, 3.0])
    assert a[1]["angle"] == pytest.approx(15.0)
    assert np.allclose(a[2]["points"], [[1, 1], [8, 2], [5, 9]])
    assert E.area_mask((32, 32), a).any()


def test_exclude_flag_and_delete(dlg):
    dlg.add_area("rect", [0.0, 10.0, 0.0, 10.0, 0.0])
    dlg.add_area("ellipse", {"center": [5.0, 5.0], "radii": [2.0, 2.0], "angle": 0.0})
    dlg.set_exclude(1, True)
    a = dlg.areas()
    assert a[1]["exclude"] is True and not a[0].get("exclude")
    dlg.delete_area(0)
    assert [x["kind"] for x in dlg.areas()] == ["ellipse"]


def test_dp_sources_disable_recentring(dlg):
    dlg.set_source("dp_mean")
    assert not dlg._chk_recenter.isEnabled()
    dlg.set_source("area")
    assert dlg._chk_recenter.isEnabled() and dlg._chk_recenter.isChecked()


def test_build_fit_apply_verify(dlg, monkeypatch):
    calls = {}

    def fake_source(sc, source, *, areas=None, recenter=True, log=None, **k):
        calls["source"] = (sc.name, source, areas, recenter)
        return dict(image=_ring(), center=(63.6, 64.3), n_patterns=10, recentered=recenter,
                    source=source, origin_source="beam_com")

    def fake_verify(sc, areas, applied_p, r_range, method="amorphous", **k):
        calls["verify"] = (sc.name, areas, tuple(applied_p), tuple(r_range), method)
        return dict(measured_a_over_b=1.03, measured_theta_deg=25.0, residual_a_over_b=1.0005,
                    residual_theta_deg=10.0, ok=True, tol=0.003)

    monkeypatch.setattr(E, "ellipse_source_image", fake_source)
    monkeypatch.setattr(E, "verify_area_ellipse", fake_verify)
    dlg.add_area("rect", [0.0, 10.0, 0.0, 31.0, 0.0])
    dlg.build_image()
    assert calls["source"][1] == "area" and calls["source"][2][0]["kind"] == "rect"
    r0, r1 = dlg.ring_range()
    assert r0 < R_RING < r1                                  # suggested from the profile
    dlg.fit()
    assert dlg._fit is not None and dlg._fit["a_over_b"] == pytest.approx(1.03, abs=0.003)
    assert "1.03" in dlg._status.text()
    dlg.set_targets(["A", "C"])
    dlg.apply()
    h = dlg._host
    assert h._scans[0].params.ellipse_source == "area" and h._scans[2].params.ellipse_source == "area"
    assert h._scans[1].params.ellipse_source == "bvm"       # not ticked
    assert h._scans[0].params.ellipse_source_file == "A"
    dlg.verify("B")
    assert calls["verify"][0] == "B" and calls["verify"][1][0]["kind"] == "rect"
    assert "OK" in dlg._status.text() or "1.0005" in dlg._status.text()
