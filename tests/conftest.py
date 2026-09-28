import sys
from pathlib import Path

# Fast4D has no package layout / setup.py — tests import top-level modules
# (pipeline.py, engine.py, ...) directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(scope="session")
def qapp():
    """Shared QApplication for tests that construct Qt widgets off-screen."""
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# ── mini synthetic Si [110] (tests/synthetic_si110.py) ─────────────────────────
# Computed once per session (~20 s each: calibration + strain without / with ROI).
# Treat them as READ-ONLY: several tests share the same Scan.

MINI_N = 24
MINI_VACUUM_COLS = 6                 # vacuum band: scan columns 0-5
MINI_ROI_EDGE = [2, 12, 8, 16]       # [x0, x1, y0, y1]: 4 of its 10 columns are vacuum
MINI_ROI_CENTER = [8, 16, 8, 16]     # all crystal, around the zero-strain line


def _mini_flow(vacuum_cols: int, roi):
    import matplotlib
    matplotlib.use("Agg")
    import synthetic_si110 as S
    vac = S.vacuum_columns(MINI_N, vacuum_cols) if vacuum_cols else None
    bv, truth = S.make_braggvectors(MINI_N, vacuum=vac)
    sc = S.make_scan(bv, roi_bounds=roi)
    logs: list[str] = []
    S.run_flow(sc, log=logs.append)
    return {"scan": sc, "truth": truth, "log": logs, "roi": roi}


@pytest.fixture(scope="session")
def mini_vacuum():
    """Mini synthetic with a vacuum band; calibration ROI crosses its edge."""
    return _mini_flow(MINI_VACUUM_COLS, MINI_ROI_EDGE)


@pytest.fixture(scope="session")
def mini_base():
    """Mini synthetic, crystal everywhere (no vacuum); central reference ROI."""
    return _mini_flow(0, MINI_ROI_CENTER)
