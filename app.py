"""fast4d.app — entrypoint for the unified Fast4D GUI (PySide6 / Qt6).

Launch (Windows, project conda env)::

    cd C:\\Users\\jtapiaca.ASURITE\\Fast4d
    conda run -n py4dstem-01419 python app.py

The heavy work runs off the GUI thread (see ``driver`` / ``qt_main``); this only
creates the QApplication, sets the icon, shows the dockable main window, and
enters the Qt event loop. Qt6 handles HiDPI scaling automatically.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

from _bootstrap import APP_ROOT, bootstrap_sys_path, icon_path

bootstrap_sys_path()
_HERE = APP_ROOT

# Blue palette — TARGETED selectors only (no blanket QWidget) so pyqtgraph image
# views, matplotlib canvases and the dark console keep their own backgrounds.
_BLUE_QSS = """
QMainWindow, QDialog { background:#EAF4FF; }
QMainWindow::separator { background:#90CAF9; width:3px; height:3px; }
QDockWidget { color:#0D47A1; }
QDockWidget::title { background:#BBDEFB; padding:5px; color:#0D47A1; font-weight:bold; }
QToolBar { background:#D6ECFF; border:none; spacing:4px; padding:3px; }
QMenuBar { background:#BBDEFB; color:#0D47A1; }
QMenuBar::item:selected { background:#64B5F6; color:white; }
QMenu { background:white; color:#0D2A4A; }
QMenu::item:selected { background:#BBDEFB; }
QTabWidget::pane { border:1px solid #90CAF9; background:#F7FBFF; }
QTabBar::tab { background:#E3F2FD; color:#0D47A1; padding:6px 12px;
               border:1px solid #90CAF9; border-bottom:none; margin-right:1px; }
QTabBar::tab:selected { background:#1976D2; color:white; }
QTabBar::tab:hover { background:#BBDEFB; }
QGroupBox { border:1px solid #90CAF9; border-radius:5px; margin-top:8px; }
QGroupBox::title { color:#0D47A1; subcontrol-origin:margin; left:8px; }
QPushButton { background:#E3F2FD; color:#0D47A1; border:1px solid #1565C0;
              border-radius:5px; padding:4px 10px; }
QPushButton:hover { background:#BBDEFB; }
QPushButton:pressed { background:#90CAF9; }
QPushButton:disabled { color:#9fb0c0; border-color:#cdd9e5; background:#eef4fb; }
QHeaderView::section { background:#BBDEFB; color:#0D47A1; border:none; padding:3px; }
QStatusBar { background:#D6ECFF; color:#0D47A1; }
QScrollArea { background:#F7FBFF; border:none; }
"""


def _install_warning_filters() -> None:
    """Keep the .bat console focused on actionable errors."""
    os.environ.setdefault("MPLBACKEND", "Agg")      # no pyplot GUI from worker threads
    warnings.filterwarnings(
        "ignore",
        message=r".*cupyx\.jit\.rawkernel is experimental.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Starting a Matplotlib GUI outside of the main thread will likely fail.*",
        category=UserWarning,
    )


def _install_qt_message_filter(QtCore) -> None:
    """Suppress one noisy Qt font warning without hiding other Qt diagnostics."""
    def handler(mode, context, message):
        if "QFont::setPointSize: Point size <= 0" in str(message):
            return
        try:
            sys.__stderr__.write(str(message) + "\n")
        except Exception:
            pass

    QtCore.qInstallMessageHandler(handler)


def main() -> int:
    _install_warning_filters()
    from PySide6 import QtCore, QtGui, QtWidgets
    _install_qt_message_filter(QtCore)

    # Throttle tqdm BEFORE py4DSTEM imports it — discards the \r bar spam (which
    # otherwise floods the cmd console and freezes the window during heavy loads)
    # and emits ~1/s progress lines into the GUI console + progress bar instead.
    import qt_tqdm
    qt_tqdm.install()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Fast4D")
    f = app.font()
    if f.pointSize() <= 0 and f.pixelSize() <= 0:
        f.setPointSize(9)
        app.setFont(f)
    app.setStyleSheet(_BLUE_QSS)               # blue palette across the GUI
    # software icon = the py4DSTEM logo (the fast-mode's icon); fall back to strain.png
    for name in ("4dstem_hero.png", "strain.png", "py4dstem_logo.png"):
        p = icon_path(name)
        if p.is_file():
            app.setWindowIcon(QtGui.QIcon(str(p)))
            break

    # Loading banner while the main window is built; heavy libs load in background.
    from qt_splash import StartupSplash
    splash = StartupSplash()
    _keep: dict = {}                       # keep refs alive past on_ready's scope

    def on_ready() -> None:
        # ALWAYS tear the splash down (finally) so a failed/slow window build can't
        # leave the frameless splash stuck in front of the desktop.
        try:
            splash.note("Building interface …")
            from qt_main import Fast4DWindow
            win = Fast4DWindow()
            _keep["win"] = win
            win.show()
            win.raise_()
            win.activateWindow()
            # Show Quick Start guide on first launch (no-op on subsequent launches).
            from qt_quickstart import QuickStartDialog
            QuickStartDialog.maybe_show(win)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            splash.close()

    splash.finished_warmup.connect(on_ready)
    splash.show()
    QtCore.QTimer.singleShot(0, splash.start)   # start once the event loop is running
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
