from types import SimpleNamespace

import engine as E


def _fake_scan(**heavy_attrs):
    defaults = dict(datacube=None, visualcube=None, vacuumcube=None, bvm_raw=None,
                    bvm_centered=None, dp_mean=None, dp_max=None, strainmap_full=None,
                    selected_disks=None, probe=None, braggpeaks=None)
    defaults.update(heavy_attrs)
    return SimpleNamespace(state=SimpleNamespace(**defaults))


def test_free_memory_nulls_heavy_buffers_but_keeps_braggpeaks_by_default():
    sc = _fake_scan(datacube="cube", probe="probe", braggpeaks="peaks")
    res = E.free_memory([sc])
    assert sc.state.datacube is None
    assert sc.state.probe is None
    # braggpeaks are the compact Path-A layer — preserved unless explicitly dropped.
    assert sc.state.braggpeaks == "peaks"
    assert res["buffers"] >= 2


def test_free_memory_drop_braggpeaks_also_releases_peaks():
    sc = _fake_scan(datacube="cube", braggpeaks="peaks")
    E.free_memory([sc], drop_braggpeaks=True)
    assert sc.state.datacube is None
    assert sc.state.braggpeaks is None


def test_free_memory_handles_empty_or_none_input():
    assert E.free_memory([])["buffers"] == 0
    assert E.free_memory(None)["buffers"] == 0


def test_free_memory_skips_scans_with_no_state():
    sc = SimpleNamespace(state=None)
    assert E.free_memory([sc])["buffers"] == 0
