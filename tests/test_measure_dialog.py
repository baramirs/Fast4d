"""Line / ROI tool window (offscreen): delete isolation, width limit, panel edits,
rotated ROI state, Ctrl+C text."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

import engine as E


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Console:
    def log(self, *_a, **_k):
        pass


class _Host(QtWidgets.QWidget):
    def __init__(self, scans):
        super().__init__()
        self._scans = scans
        self._active = 0
        self._console = _Console()


@pytest.fixture
def dlg(app, monkeypatch):
    import qt_measure as Q
    ramp = np.tile(np.arange(80, dtype=float), (60, 1))
    monkeypatch.setattr(E, "channel_map_2d", lambda sc, ch, lab: ramp)
    scans = [E.Scan(name="a"), E.Scan(name="b")]
    d = Q.MeasureDialog(_Host(scans))
    yield d
    d.close()


def test_opens_with_one_line_and_arrow(dlg):
    assert [e.id for e in dlg._model.elements] == ["L1"]
    assert len(dlg._items["L1"]["deco"]) == 3             # band, start bar, arrowhead


def test_delete_selected_keeps_the_others(dlg):
    dlg._add_line()
    dlg._add_roi()
    g2 = [list(p) for p in dlg._model.get("L2").geom]
    dlg._select("L1")
    dlg._delete_selected()
    assert [e.id for e in dlg._model.elements] == ["L2", "R1"]
    assert dlg._model.get("L2").geom == g2
    assert set(dlg._items) == {"L2", "R1"}


def test_width_spin_limited_to_200(dlg):
    assert dlg._lw.maximum() == 200
    dlg._lw.setValue(500)
    assert dlg._model.get("L1").width == 200


def test_panel_edit_moves_the_line(dlg):
    dlg._lx0.setValue(10.0)
    dlg._ly0.setValue(20.0)
    assert dlg._model.get("L1").geom[0] == [10.0, 20.0]
    g = dlg._line_geom(dlg._items["L1"]["roi"])
    assert np.allclose(g[0], [10.0, 20.0])


def test_rotated_roi_state_roundtrip(dlg):
    dlg._add_roi()
    el = dlg._model.selected_element()
    dlg._rang.setValue(30.0)
    g = dlg._rect_geom(dlg._items[el.id]["roi"])
    assert np.isclose(g[4], 30.0)
    assert np.allclose(g[:4], el.geom[:4], atol=1e-6)


def test_ctrl_c_text_profile(dlg):
    dlg._update_plots()
    lines = dlg.clipboard_text().splitlines()
    assert lines[0] == "distance (px)\tL1 a (%)"
    assert len(lines) > 2
