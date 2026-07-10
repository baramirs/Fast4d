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
