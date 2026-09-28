"""Report > Session analysis acts only on the files ticked under Files + Refresh."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6 import QtWidgets

import engine as E
import session_scope as SS


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(app, monkeypatch):
    import analysis as A
    import qt_report as R

    SS.reset()
    seen = []

    def fake_stats(scans, label):
        seen.append([s.name for s in scans])
        return pd.DataFrame({"scan": [s.name for s in scans]})

    monkeypatch.setattr(A, "cross_scan_stats", fake_stats)
    scans = [E.Scan(name="a"), E.Scan(name="b"), E.Scan(name="c")]
    p = R.ReportPanel(lambda: scans)
    p._seen = seen
    return p


def _select_action(panel, action):
    import qt_report as R
    root = panel._tree.invisibleRootItem()
    for i in range(root.childCount()):
        it = root.child(i)
        if (it.data(0, R._ROLE) or {}).get("kind") != "session_root":
            continue
        for j in range(it.childCount()):
            ch = it.child(j)
            if (ch.data(0, R._ROLE) or {}).get("action") == action:
                panel._tree.setCurrentItem(ch)
                panel._show_current()
                return
    raise AssertionError(f"action {action} not in tree")


def test_session_actions_trimmed():
    import qt_report as R
    actions = [a for _t, a in R._SESSION_ACTIONS]
    assert actions == ["dist", "corr", "stats", "stress", "calib_plot", "calib_table"]


def test_nothing_shown_without_refresh(panel):
    panel._tick_session_files(["a", "b"])
    _select_action(panel, "stats")
    assert panel._seen == []
    assert "Refresh" in panel._status.text()


def test_refresh_feeds_only_ticked_files(panel):
    panel._tick_session_files(["a", "c"])
    panel._refresh_session_files()
    _select_action(panel, "stats")
    assert panel._seen[-1] == ["a", "c"]


def test_ticking_without_refresh_keeps_previous_set(panel):
    panel._tick_session_files(["b"])
    panel._refresh_session_files()
    panel._tick_session_files(["a", "b"])
    _select_action(panel, "stats")
    assert panel._seen[-1] == ["b"]
    assert "pending" in panel._files_label.text()
