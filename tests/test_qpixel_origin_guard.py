"""Q-pixel fit requires Origin on braggpeaks (py4DSTEM center=True)."""
from __future__ import annotations

import pytest


def test_braggpeaks_has_origin_false_without_bp():
    import engine as E

    class S:
        name = "t"
        state = type("St", (), {"braggpeaks": None})()
        params = type("P", (), {"center_guess": [128.0, 128.0]})()

    assert E.braggpeaks_has_origin(S()) is False


def test_braggpeaks_has_origin_reads_calstate():
    import engine as E

    class BP:
        calibration = None
        calstate = {"center": True, "ellipse": False, "pixel": False, "rotate": False}

    class S:
        name = "t"
        state = type("St", (), {"braggpeaks": BP()})()

    assert E.braggpeaks_has_origin(S()) is True


def test_ensure_origin_raises_without_center_guess():
    import engine as E

    class BP:
        calibration = None
        calstate = {"center": False}

    class St:
        braggpeaks = BP()

    class S:
        name = "demo"
        state = St()
        params = type("P", (), {"center_guess": None})()

        def ensure_state(self):
            return self.state

    with pytest.raises(RuntimeError, match="Origin first"):
        E.ensure_origin_for_qpixel(S(), log=None)
