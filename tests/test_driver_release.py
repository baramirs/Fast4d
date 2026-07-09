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
