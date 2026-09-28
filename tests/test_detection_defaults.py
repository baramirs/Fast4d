"""D1: default detection / Q-pixel settings see the high-order reflections.

With minRelativeIntensity 0.05 and q_kmax 1.0 only 111 / 220 were detected. Measured on
Scan01 (route B, PARA TEST, 28-09-2026): Q +57 % off the seed, BVM indexing 12.5 % off
in d, strain frame turned 35 deg, strain ~0.9 % RMS off route A. With 0.001 / 1.6 (the
notebook's detection): Q 0.021568, axis within 0.02 deg, p95 of the differences <= 0.03 %.
Synthetic Si [110]: Q +0.153 % -> +0.026 %, eyy RMS 0.122 % -> 0.070 %.
"""
import engine as E
from state import WorkflowState


def test_calibration_params_defaults():
    p = E.CalibrationParams()
    assert p.detect_min_relative_intensity == 0.001
    assert p.q_kmax == 1.6


def test_pipeline_detect_params_default():
    assert WorkflowState().detect_params["minRelativeIntensity"] == 0.001
