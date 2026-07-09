from types import SimpleNamespace

import engine as E


def _fake_scan(datacube=None):
    return SimpleNamespace(state=SimpleNamespace(
        datacube=datacube, visualcube=None, vacuumcube=None, bvm_raw=None,
        bvm_centered=None, dp_mean=None, dp_max=None, strainmap_full=None,
        selected_disks=None, probe=None,
    ))


def test_default_policy_keeps_two_most_recent():
    assert E.get_data_policy().max_scans_in_ram == 2


def test_set_data_policy_updates_the_shared_instance():
    E.set_data_policy(max_scans_in_ram=3)
    try:
        assert E.get_data_policy().max_scans_in_ram == 3
    finally:
        E.set_data_policy(max_scans_in_ram=2)  # restore default for other tests


def test_enforce_resident_data_limit_keeps_active_plus_window_releases_rest():
    E.set_data_policy(max_scans_in_ram=2)
    scans = [_fake_scan(datacube=f"cube_{i}") for i in range(4)]

    # Eviction is unconditional (see test_..._releases_on_the_very_first_call): a scan
    # that hasn't been visited yet is not "recent" and gets released immediately, even
    # if it happened to already hold data (e.g. from an earlier batch compute). Each
    # scan below has its data re-populated right before it becomes active, simulating
    # the reload a real scan-select would trigger.
    recent = E.enforce_resident_data_limit(scans, active_index=0, recent_indices=[])
    assert recent == [0]
    assert scans[0].state.datacube == "cube_0"  # active scan untouched
    assert scans[1].state.datacube is None       # released — never yet visited
    assert scans[2].state.datacube is None
    assert scans[3].state.datacube is None

    scans[2].state.datacube = "cube_2"  # simulate reload before this scan becomes active
    recent = E.enforce_resident_data_limit(scans, active_index=2, recent_indices=recent)
    assert recent == [2, 0]
    assert scans[2].state.datacube == "cube_2"       # newly active, untouched
    assert scans[0].state.datacube == "cube_0"        # still in the 2-item window
    assert scans[1].state.datacube is None            # released — never in the window
    assert scans[3].state.datacube is None            # released — never in the window

    recent = E.enforce_resident_data_limit(scans, active_index=1, recent_indices=recent)
    assert recent == [1, 2]
    assert scans[0].state.datacube is None  # fell out of the window this time


def test_enforce_resident_data_limit_respects_a_larger_configured_window():
    E.set_data_policy(max_scans_in_ram=3)
    try:
        scans = [_fake_scan(datacube=f"cube_{i}") for i in range(4)]
        # As above: eviction is unconditional, so re-populate each scan's data right
        # before it becomes active, simulating a reload triggered by selecting it.
        recent = E.enforce_resident_data_limit(scans, active_index=0, recent_indices=[])
        scans[1].state.datacube = "cube_1"
        recent = E.enforce_resident_data_limit(scans, active_index=1, recent_indices=recent)
        scans[2].state.datacube = "cube_2"
        recent = E.enforce_resident_data_limit(scans, active_index=2, recent_indices=recent)
        assert recent == [2, 1, 0]
        assert all(scans[i].state.datacube == f"cube_{i}" for i in range(3))
    finally:
        E.set_data_policy(max_scans_in_ram=2)


def test_enforce_resident_data_limit_releases_on_the_very_first_call():
    E.set_data_policy(max_scans_in_ram=2)
    scans = [_fake_scan(datacube=f"cube_{i}") for i in range(10)]
    recent = E.enforce_resident_data_limit(scans, active_index=0, recent_indices=[])
    assert recent == [0]
    assert scans[0].state.datacube == "cube_0"  # active, untouched
    for i in range(1, 10):
        assert scans[i].state.datacube is None  # all released immediately, no ramp-up delay


def test_enforce_resident_data_limit_shrinks_when_policy_limit_lowered():
    E.set_data_policy(max_scans_in_ram=3)
    try:
        scans = [_fake_scan(datacube=f"cube_{i}") for i in range(4)]
        # Eviction is unconditional, so re-populate each scan's data right before it
        # becomes active, simulating a reload triggered by selecting it.
        recent = E.enforce_resident_data_limit(scans, active_index=0, recent_indices=[])
        scans[1].state.datacube = "cube_1"
        recent = E.enforce_resident_data_limit(scans, active_index=1, recent_indices=recent)
        scans[2].state.datacube = "cube_2"
        recent = E.enforce_resident_data_limit(scans, active_index=2, recent_indices=recent)
        assert recent == [2, 1, 0]
        E.set_data_policy(max_scans_in_ram=1)
        recent = E.enforce_resident_data_limit(scans, active_index=2, recent_indices=recent)
        assert recent == [2]
        assert scans[2].state.datacube == "cube_2"   # still active, untouched
        assert scans[1].state.datacube is None        # fell out when window shrank to 1
        assert scans[0].state.datacube is None        # fell out when window shrank to 1
    finally:
        E.set_data_policy(max_scans_in_ram=2)
