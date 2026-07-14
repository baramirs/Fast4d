from types import SimpleNamespace

import driver
import engine as E


def test_compute_all_releases_memory_after_each_scan(monkeypatch):
    scans = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    fake_outcome = SimpleNamespace(ok=True, error=None, elapsed_s=0.1)
    monkeypatch.setattr(driver, "compute_scan", lambda *a, **k: fake_outcome)

    release_calls = []
    monkeypatch.setattr(E, "free_memory", lambda scans_arg, **k: release_calls.append(list(scans_arg)))

    driver.compute_all(scans)

    assert release_calls == [[scans[0]], [scans[1]]]


def test_compute_all_still_calls_on_scan_done_before_releasing(monkeypatch):
    scans = [SimpleNamespace(name="a")]
    fake_outcome = SimpleNamespace(ok=True, error=None, elapsed_s=0.1)
    monkeypatch.setattr(driver, "compute_scan", lambda *a, **k: fake_outcome)
    order = []
    monkeypatch.setattr(E, "free_memory", lambda *a, **k: order.append("release"))

    driver.compute_all(scans, on_scan_done=lambda out: order.append("on_scan_done"))

    assert order == ["on_scan_done", "release"]


def test_compute_all_drops_braggpeaks_when_path_persisted(monkeypatch, tmp_path):
    bp = tmp_path / "braggpeaks.h5"
    bp.write_bytes(b"x")  # a real on-disk file → peaks are reloadable
    scans = [SimpleNamespace(name="a", braggpeaks_path=str(bp)),
             SimpleNamespace(name="b", braggpeaks_path=None)]
    fake_outcome = SimpleNamespace(ok=True, error=None, elapsed_s=0.1)
    monkeypatch.setattr(driver, "compute_scan", lambda *a, **k: fake_outcome)

    drop_flags = []
    monkeypatch.setattr(E, "free_memory",
                        lambda scans_arg, **k: drop_flags.append(k.get("drop_braggpeaks")))

    driver.compute_all(scans)

    # scan a has a persisted .h5 → drop; scan b has none → keep.
    assert drop_flags == [True, False]
