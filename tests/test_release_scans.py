from types import SimpleNamespace

import engine as E


def _fake_scan(**heavy_attrs):
    defaults = dict(datacube=None, visualcube=None, vacuumcube=None, bvm_raw=None,
                     bvm_centered=None, dp_mean=None, dp_max=None, strainmap_full=None,
                     selected_disks=None, probe=None)
    defaults.update(heavy_attrs)
    return SimpleNamespace(state=SimpleNamespace(**defaults))


def test_release_scans_nulls_heavy_attrs_and_counts_them():
    sc = _fake_scan(datacube="fake_cube", probe="fake_probe")
    n = E.release_scans([sc])
    assert n == 2
    assert sc.state.datacube is None
    assert sc.state.probe is None


def test_release_scans_skips_scans_with_no_state():
    sc = SimpleNamespace(state=None)
    assert E.release_scans([sc]) == 0


def test_release_scans_handles_empty_or_none_input():
    assert E.release_scans([]) == 0
    assert E.release_scans(None) == 0


def test_release_scans_does_not_touch_figures_or_braggpeaks():
    sc = _fake_scan(datacube="fake_cube")
    sc.state.figures = {"origin": "fake_fig"}
    sc.state.braggpeaks = "fake_peaks"
    E.release_scans([sc])
    assert sc.state.figures == {"origin": "fake_fig"}
    assert sc.state.braggpeaks == "fake_peaks"
