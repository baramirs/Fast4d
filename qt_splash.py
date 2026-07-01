"""fast4d.qt_splash — startup loading banner + deferred heavy imports.

Heavy libraries (py4DSTEM, CuPy, …) cost several seconds on a cold start. We show
the main window first (~2–3 s), then warm them in a background thread so launch
feels snappy. Compute / calibration waits briefly if the user clicks before warm-up
finishes.
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from _bootstrap import APP_ROOT, bootstrap_sys_path, icon_path

bootstrap_sys_path()
_HERE = APP_ROOT

# Loaded in the background after the main window is visible.
HEAVY_WARMUP_STEPS: list[tuple[str, str]] = [
    ("matplotlib", "matplotlib.pyplot"),
    ("py4DSTEM", "py4DSTEM"),
    ("CuPy (GPU)", "cupy"),
    ("Pipeline", "pipeline"),
]

_heavy_ready = threading.Event()
_heavy_started = False
_heavy_lock = threading.Lock()


def heavy_ready() -> bool:
    return _heavy_ready.is_set()


def ensure_heavy_imports(*, log=None, block: bool = False) -> bool:
    """Start background warm-up if needed; optionally block until ready."""
    start_heavy_warmup(log=log)
    if block:
        _heavy_ready.wait()
    return heavy_ready()


def start_heavy_warmup(*, log=None) -> None:
    """Kick off a single background import of py4DSTEM / CuPy / pipeline."""
    global _heavy_started
    with _heavy_lock:
        if _heavy_started or _heavy_ready.is_set():
            return
        _heavy_started = True
    threading.Thread(
        target=_heavy_worker, args=(log,), daemon=True, name="heavy-warmup",
    ).start()


def _heavy_worker(log) -> None:
    t0 = time.perf_counter()
    for label, module in HEAVY_WARMUP_STEPS:
        _emit(log, f"Loading {label} …")
        try:
            importlib.import_module(module)
            _emit(log, f"   ok  {label}")
        except Exception as exc:
            _emit(log, f"   --  {label} unavailable: {exc}")
    _emit(log, f"Heavy components ready in {time.perf_counter() - t0:0.1f}s.")
    _heavy_ready.set()


def _emit(log, msg: str) -> None:
    if log is not None:
        try:
            log(msg)
        except Exception:
            pass


class StartupSplash(QtWidgets.QDialog):
    """Brief frameless banner while the main window is built (no heavy imports)."""

    finished_warmup = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setObjectName("StartupSplash")
        self.resize(420, 120)
        self.setStyleSheet("QDialog#StartupSplash{background:#0D47A1;}")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        header = QtWidgets.QHBoxLayout()
        logo_path = icon_path("4dstem_hero.png")
        if not logo_path.is_file():
            logo_path = icon_path("py4dstem_logo.png")
        if logo_path.is_file():
            logo = QtWidgets.QLabel()
            pm = QtGui.QPixmap(str(logo_path))
            if not pm.isNull():
                logo.setPixmap(pm.scaledToHeight(
                    48, QtCore.Qt.TransformationMode.SmoothTransformation))
            header.addWidget(logo)
        titles = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Fast4D")
        title.setStyleSheet("color:#FFFFFF; font-size:26px; font-weight:700;")
        sub = QtWidgets.QLabel("Starting…")
        sub.setStyleSheet("color:#BBDEFB; font-size:11px;")
        self._sub = sub
        titles.addWidget(title)
        titles.addWidget(sub)
        header.addLayout(titles)
        header.addStretch(1)
        lay.addLayout(header)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 0)          # indeterminate
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            "QProgressBar{border:1px solid #1565C0; border-radius:4px; background:#1565C0;"
            "height:12px;}"
            "QProgressBar::chunk{background:#64B5F6; border-radius:3px;}")
        lay.addWidget(self._bar)
        self._started = False

    def note(self, msg: str) -> None:
        self._sub.setText(msg)
        QtWidgets.QApplication.processEvents()

    def start(self) -> None:
        """Emit immediately — heavy imports run later in the background."""
        if self._started:
            return
        self._started = True
        self.finished_warmup.emit()
