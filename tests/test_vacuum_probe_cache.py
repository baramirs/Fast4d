"""Vacuum-probe cache + alpha-based detector defaults (no py4DSTEM needed)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as pl  # noqa: E402
import engine as E  # noqa: E402
from state import WorkflowState  # noqa: E402


class _FakeProbe:
    def __init__(self, img):
        self.probe = img
        self.kernel = None

    def get_kernel(self, mode, radii):
        self.kernel = self.probe - self.probe.mean()


class _FakeCube:
    Rshape = (8, 8)

    def __init__(self, a=12.0):
        yy, xx = np.mgrid[:64, :64]
        self.img = ((yy - 31.0) ** 2 + (xx - 33.0) ** 2 < a * a).astype(np.float32)
        self.a = a

    def get_vacuum_probe(self, ROI):
        return _FakeProbe(self.img)

    def get_probe_size(self, arr):
        return self.a, 31.0, 33.0


@pytest.fixture
def fake(monkeypatch, tmp_path):
    loads = []

    def _load(path, log=None):
        loads.append(str(path))
        return _FakeCube()

    monkeypatch.setattr(pl, "_load_raw_datacube", _load)
    monkeypatch.setattr(pl, "_with_plot_suppressed", lambda f: f())
    monkeypatch.setenv("FAST4D_PROBE_CACHE", str(tmp_path / "cache"))
    pl.clear_vacuum_probe_cache()
    vac = tmp_path / "vacuum.mib"
    vac.write_bytes(b"x" * 100)
    yield vac, loads
    pl.clear_vacuum_probe_cache()


def test_vacuum_loaded_once_for_many_scans(fake):
    vac, loads = fake
    states = [WorkflowState() for _ in range(4)]
    for st in states:
        pl.compute_probe_step(st, vac)
    assert len(loads) == 1                         # one vacuum load for 4 scans
    assert all(st.probe is states[0].probe for st in states)   # shared object
    assert all(st.vacuumcube is None for st in states)         # 4D not kept
    assert states[2].probe_alpha == pytest.approx(12.0)


def test_disk_cache_skips_vacuum_in_new_session(fake):
    vac, loads = fake
    pl.compute_probe_step(WorkflowState(), vac)
    pl.clear_vacuum_probe_cache()                  # "new session"
    st = WorkflowState()
    pl.compute_probe_step(st, vac)
    assert len(loads) == 1
    k = pl.probe_kernel_template_ndarray(st.probe)
    assert k.shape == (64, 64) and np.isfinite(k).all()
    assert st.probe_qx0 == pytest.approx(31.0)


def test_changed_vacuum_file_invalidates(fake):
    vac, loads = fake
    pl.compute_probe_step(WorkflowState(), vac)
    vac.write_bytes(b"y" * 120)
    os.utime(vac, None)
    pl.compute_probe_step(WorkflowState(), vac)
    assert len(loads) == 2


def test_detector_radii_defaults_and_clamp():
    r = E.detector_radii_from_alpha(10.0)
    assert (r["bf"], r["adf_inner"], r["adf_outer"]) == (10.0, 20.0, 80.0)
    r = E.detector_radii_from_alpha(20.0, center_yx=(128, 128), qshape=(256, 256))
    assert r["clamped"] and r["adf_outer"] == pytest.approx(127.0)
    r = E.detector_radii_from_alpha(20.0, center_yx=(128, 128), qshape=(256, 256),
                                    clamp=False)
    assert not r["clamped"] and r["adf_outer"] == 160.0
