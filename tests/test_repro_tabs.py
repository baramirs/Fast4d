"""Box/violin and PCA views live in the reproducibility plugin (moved out of the Report)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Host(QtWidgets.QWidget):
    def __init__(self, scans):
        super().__init__()
        self._scans = scans
        self._active = 0


def _dialog(scans):
    import qt_main as M
    return M.ReproducibilityDialog(_Host(scans))


def test_tabs_present(app):
    import engine as E
    d = _dialog([E.Scan(name="a"), E.Scan(name="b")])
    titles = [d._tabs.tabText(i) for i in range(d._tabs.count())]
    assert "Distributions (box / violin)" in titles
    assert "PCA of scans" in titles


def test_scan_views_use_the_run_scans(app, monkeypatch):
    import analysis as A
    import engine as E
    from matplotlib.figure import Figure

    calls = []
    monkeypatch.setattr(A, "boxplot_figure",
                        lambda scans, label, kind: calls.append(("box", [s.name for s in scans],
                                                                  label, kind)) or Figure())
    monkeypatch.setattr(A, "pca_figure",
                        lambda scans, label: calls.append(("pca", [s.name for s in scans],
                                                           label)) or Figure())
    scans = [E.Scan(name="a"), E.Scan(name="b"), E.Scan(name="c")]
    d = _dialog(scans)
    d._run_scans = scans[:2]
    d._show_scan_views("with_roi")
    assert ("box", ["a", "b"], "with_roi", "violin") in calls
    assert ("pca", ["a", "b"], "with_roi") in calls
    assert d._box_host.count() > 0 and d._pca_host.count() > 0
