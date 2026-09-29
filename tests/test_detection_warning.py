"""D1: warn when the detection setting keeps only the low-order disks.

minRelativeIntensity is the user's choice (Detection tab), so Fast4D keeps its value and
warns. Measured on Scan01 route B (PARA TEST, 28-09-2026): 0.05 kept 6.9 peaks / pattern
(mostly 111, 220); Q-pixel fit +57 % off the seed, BVM indexing 12.5 % off in d, strain
frame turned 35 deg. With 0.001 (the notebook's value): 24.7 peaks / pattern, Q 0.021568.
"""
import engine as E


def _scan(minrel):
    sc = E.Scan(name="s")
    sc.params.detect_min_relative_intensity = minrel
    return sc


def test_high_min_relative_intensity_warns():
    w = E.detection_param_warnings(_scan(0.05).params)
    assert len(w) == 1 and "minRelativeIntensity" in w[0] and "0.05" in w[0]


def test_notebook_value_does_not_warn():
    assert E.detection_param_warnings(_scan(0.001).params) == []


def test_compute_braggpeaks_logs_the_warning(monkeypatch):
    monkeypatch.setattr(E._pipeline(), "compute_braggpeaks_step",
                        lambda state, save_path=None, log=None: None)
    logs = []
    E.compute_braggpeaks(_scan(0.05), log=logs.append)
    assert any(s.startswith("WARNING") and "minRelativeIntensity" in s for s in logs)
    logs.clear()
    E.compute_braggpeaks(_scan(0.001), log=logs.append)
    assert not any("WARNING" in s for s in logs)
