from types import SimpleNamespace

import engine as E


def test_drop_then_ensure_reloads_from_disk(monkeypatch):
    """Fase 2 contract: free_memory(drop_braggpeaks=True) releases the peaks, and
    the light Path-A entry ensure_braggpeaks_for_calibration re-loads them on demand
    from the saved .h5 — so dropping resident peaks never loses data."""
    state = SimpleNamespace(braggpeaks="peaks-in-ram")
    scan = SimpleNamespace(state=state, ensure_state=lambda: state,
                           braggpeaks_path="C:/somewhere/braggpeaks.h5")

    # Drop the resident peaks (as Free RAM / batch does when a .h5 exists).
    E.free_memory([scan], drop_braggpeaks=True)
    assert state.braggpeaks is None

    # ensure_braggpeaks_for_calibration must trigger a reload from disk (Path A).
    monkeypatch.setattr(E, "analysis_path", lambda sc: "A")

    def _fake_load(sc, *, log=None):
        sc.ensure_state().braggpeaks = "peaks-reloaded"

    monkeypatch.setattr(E, "load_braggpeaks", _fake_load)

    E.ensure_braggpeaks_for_calibration(scan)
    assert state.braggpeaks == "peaks-reloaded"


def test_ensure_is_noop_when_peaks_already_resident(monkeypatch):
    state = SimpleNamespace(braggpeaks="still-here")
    scan = SimpleNamespace(state=state, ensure_state=lambda: state)

    called = []
    monkeypatch.setattr(E, "load_braggpeaks", lambda sc, **k: called.append(1))

    E.ensure_braggpeaks_for_calibration(scan)
    assert called == [], "must not reload when peaks are already in RAM"
    assert state.braggpeaks == "still-here"
