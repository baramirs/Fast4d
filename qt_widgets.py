"""fast4d.qt_widgets — reusable PySide6 panels for the Fast4D window.

    ResourceMonitor  — live RAM / GPU(VRAM) / CPU bars (psutil + pynvml/nvidia-smi)
    CalStateStrip    — calibration-state lights (origin…strain…stress), live colors
    ConsoleWidget    — thread-safe py4DSTEM message console
    AdfView          — pyqtgraph ADF / virtual-image viewer (pan / zoom / levels)

All are self-contained (no app_tk coupling). The calstate state-derivation reuses
``pipeline.single_scan_cal_ui_flags`` (lazy import); the dot styling mirrors the
shared ``gui_status_widgets`` palette, re-declared here to avoid importing a Tk
module into a Qt one.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


def _enable_minmax(dlg) -> None:
    """Add minimize / maximize buttons to a dialog (Qt shows only Close by default)."""
    dlg.setWindowFlags(dlg.windowFlags()
                       | QtCore.Qt.WindowType.WindowMinimizeButtonHint
                       | QtCore.Qt.WindowType.WindowMaximizeButtonHint)

# Fast4d is self-contained: pipeline.py lives alongside this file, so only the
# app root needs to be importable (the legacy ``_HERE.parent`` / home-dir
# injection was removed to avoid shadowing unrelated modules from the home dir).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


class FlowLayout(QtWidgets.QLayout):
    """Wrap widgets to the next row when the parent grows narrow (no text squish)."""

    def __init__(self, parent=None, margin=0, hspacing=6, vspacing=6) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._hspace = int(hspacing)
        self._vspace = int(vspacing)
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QtCore.QRect, *, test_only: bool) -> int:
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        row_h = 0
        max_w = rect.width() - m.left() - m.right()
        x0 = x
        for item in self._items:
            w = item.widget()
            if w is not None:
                w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                 QtWidgets.QSizePolicy.Policy.Fixed)
            hint = item.sizeHint()
            next_x = x + hint.width() + self._hspace
            if next_x - self._hspace > x0 + max_w and row_h > 0:
                x = x0
                y += row_h + self._vspace
                next_x = x + hint.width() + self._hspace
                row_h = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y() + m.bottom()


# ─────────────────────────────────────────────────────────────────────────────
# Calibration-state lights
# ─────────────────────────────────────────────────────────────────────────────

# (symbol, background, foreground) — mirrors gui_status_widgets.CALSTATE_STYLE.
CALSTATE_STYLE: dict[str, tuple[str, str, str]] = {
    "applied": ("✓", "#2E7D32", "#ffffff"),
    "staged":  ("◐", "#E65100", "#ffffff"),
    "pending": ("✗", "#C62828", "#ffffff"),
    "unused":  ("·", "#BDBDBD", "#212121"),
}
CALSTATE_ITEMS: tuple[tuple[str, str], ...] = (
    ("origin", "Origin"), ("ellipse", "Ellipse"), ("qpx", "Q-px"),
    ("basis", "Basis"), ("strain", "Strain"), ("stress", "Stress"),
    ("lines", "Lines"),
)


class CalStateStrip(QtWidgets.QWidget):
    """A row of calibration-state dots that recolor as steps complete."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dots: dict[str, QtWidgets.QLabel] = {}
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(6)
        title = QtWidgets.QLabel("Calibration")
        title.setStyleSheet("font-weight:bold;")
        lay.addWidget(title)
        for key, label in CALSTATE_ITEMS:
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(0)
            dot = QtWidgets.QLabel("✗")
            dot.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            dot.setFixedSize(24, 24)
            cap = QtWidgets.QLabel(label)
            cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            cap.setStyleSheet("font-size:8px; color:#555;")
            col.addWidget(dot)
            col.addWidget(cap)
            lay.addLayout(col)
            self._dots[key] = dot
        lay.addStretch(1)
        self.update_states({})

    def update_states(self, states: dict[str, str]) -> None:
        """Apply a {key: applied|staged|pending|unused} mapping to the dots."""
        for key, dot in self._dots.items():
            sym, bg, fg = CALSTATE_STYLE.get(states.get(key, "pending"),
                                             CALSTATE_STYLE["pending"])
            dot.setText(sym)
            dot.setStyleSheet(
                f"background:{bg}; color:{fg}; border-radius:4px; font-weight:bold;")

    def update_from_state(self, state) -> None:
        """Derive states from a WorkflowState via the proven pipeline mapper."""
        if state is None:
            self.update_states({})
            return
        try:
            from pipeline import single_scan_cal_ui_flags
            self.update_states(single_scan_cal_ui_flags(state))
        except Exception:
            self.update_states({})


# ─────────────────────────────────────────────────────────────────────────────
# Resource monitor (RAM / GPU / CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024.0 ** 3):.1f}"


def _gpu_mem_used_total() -> tuple[float, float] | None:
    """(used, total) VRAM bytes via NVML, else nvidia-smi, else None."""
    try:
        from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetMemoryInfo, nvmlShutdown)
        nvmlInit()
        try:
            info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
            return float(info.used), float(info.total)
        finally:
            try:
                nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True, timeout=1.5).strip()
            first = out.splitlines()[0]
            used_mb, total_mb = (float(p.strip()) for p in first.split(",")[:2])
            return used_mb * 1024 ** 2, total_mb * 1024 ** 2
        except Exception:
            return None
    return None


class ResourceMonitor(QtWidgets.QWidget):
    """RAM / GPU / CPU usage rows (name | value | bar), polled on a QTimer."""

    def __init__(self, parent=None, *, interval_ms: int = 800) -> None:
        super().__init__(parent)
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(4, 2, 4, 2)
        grid.setVerticalSpacing(2)
        self._bars: dict[str, QtWidgets.QProgressBar] = {}
        self._vals: dict[str, QtWidgets.QLabel] = {}
        for r, name in enumerate(("RAM", "GPU", "CPU")):
            lab = QtWidgets.QLabel(name)
            lab.setStyleSheet("font-weight:bold;")
            val = QtWidgets.QLabel("—")
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(12)
            grid.addWidget(lab, r, 0)
            grid.addWidget(val, r, 1)
            grid.addWidget(bar, r, 2)
            grid.setColumnStretch(2, 1)
            self._vals[name] = val
            self._bars[name] = bar
        self._gpu_cache = None        # (used,total) bytes; refreshed off-thread
        self._gpu_busy = False
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.start(interval_ms)
        self.tick()

    def _set(self, name: str, value_text: str, pct: float) -> None:
        self._vals[name].setText(value_text)
        pct = max(0.0, min(100.0, float(pct)))
        self._bars[name].setValue(int(pct))
        color = "#2E7D32" if pct < 70 else ("#E65100" if pct < 90 else "#C62828")
        self._bars[name].setStyleSheet(
            f"QProgressBar::chunk{{background:{color};}}")

    def tick(self) -> None:
        # RAM + CPU via psutil
        try:
            import psutil
            vm = psutil.virtual_memory()
            self._set("RAM", f"{_fmt_gb(vm.used)} / {_fmt_gb(vm.total)} GB", vm.percent)
            self._set("CPU", f"{psutil.cpu_percent():.0f} %", psutil.cpu_percent())
        except Exception:
            self._vals["RAM"].setText("(psutil?)")
        # GPU: show the CACHED value; refresh it OFF the GUI thread. nvidia-smi is a
        # subprocess that can block up to ~1.5 s (worse while CUDA is saturated mid-
        # compute) — calling it here would freeze the window every tick.
        gpu = self._gpu_cache
        if gpu is None:
            self._set("GPU", "n/a", 0.0)
        else:
            used, total = gpu
            pct = 100.0 * used / total if total else 0.0
            self._set("GPU", f"{_fmt_gb(used)} / {_fmt_gb(total)} GB", pct)
        if not self._gpu_busy:
            self._gpu_busy = True
            threading.Thread(target=self._refresh_gpu, daemon=True).start()

    def _refresh_gpu(self) -> None:
        try:
            self._gpu_cache = _gpu_mem_used_total()
        except Exception:
            self._gpu_cache = None
        finally:
            self._gpu_busy = False


# ─────────────────────────────────────────────────────────────────────────────
# Export selection — shared by Save / Save As… / Export PPTX…
# ─────────────────────────────────────────────────────────────────────────────

class ExportSelectionDialog(QtWidgets.QDialog):
    """Pick which figure/slide categories to generate or export. Data files
    (CSV/XLSX/JSON) are always written regardless of these selections — only
    figure/image rendering is gated. The same dialog is reused by Save,
    Save As… and Export PPTX…"""

    CATEGORIES = [
        ("per_scan", "Per-scan figures (probe, calibration steps, strain/stress maps, profiles…)"),
        ("lines_group", "Lines across files (grouped — overlay/points/histogram)"),
        ("rois_group", "ROIs across files (grouped — bars/points/histogram)"),
        ("repeatability", "Repeatability figures (pixel-wise differences)"),
        ("calib_trends", "Calibration trend plots (PPTX)"),
        ("strain_maps", "Strain/stress map slides (PPTX)"),
        ("basis_panels", "Basis-vector panel slides (PPTX)"),
    ]

    def __init__(self, parent=None, initial: dict | None = None, *,
                 title: str = "Choose what to export") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        lay = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Data (CSV / XLSX / JSON) is always saved.\n"
            "Choose which figures/slides to generate:")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        initial = initial or {}
        self._checks: dict[str, QtWidgets.QCheckBox] = {}
        for key, label in self.CATEGORIES:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(bool(initial.get(key, True)))
            lay.addWidget(cb)
            self._checks[key] = cb

        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Select all")
        btn_none = QtWidgets.QPushButton("Select none")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _set_all(self, checked: bool) -> None:
        for cb in self._checks.values():
            cb.setChecked(checked)

    def result_selection(self) -> dict:
        """{category_key: bool} reflecting the current checkbox states."""
        return {key: cb.isChecked() for key, cb in self._checks.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Calibration loading progress (live log while upstream steps run)
# ─────────────────────────────────────────────────────────────────────────────

class CalibLoadingDialog(QtWidgets.QDialog):
    """Small modal-less window showing calibration prep steps as they run."""

    _append = QtCore.Signal(str)

    def __init__(self, parent=None, *, title: str = "Loading calibration") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(540, 340)
        lay = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Applying previous calibration steps from the parameter table…")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#1565C0; font-size:11px;")
        lay.addWidget(hint)
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "QPlainTextEdit{background:#1E1E1E; color:#D4D4D4;"
            "font-family:Consolas,'Courier New',monospace; font-size:10px;}")
        lay.addWidget(self._log, 1)
        self._append.connect(self._append_line)

    @QtCore.Slot(str)
    def _append_line(self, msg: str) -> None:
        self._log.appendPlainText(str(msg).rstrip("\n"))
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def append_log(self, msg: str) -> None:
        """Thread-safe — safe to call from worker threads."""
        self._append.emit(str(msg))


# ─────────────────────────────────────────────────────────────────────────────
# Console (thread-safe, explicit log lines only)
# ─────────────────────────────────────────────────────────────────────────────


class ConsoleWidget(QtWidgets.QPlainTextEdit):
    """Read-only log for explicit ``log()`` calls from the pipeline / UI.

    py4DSTEM prints and tqdm progress bars stay in the terminal window that
    launched the app (``run_gui.bat``). Mirroring stdout here floods the Qt
    event loop and freezes the whole window.
    """

    _line = QtCore.Signal(str)

    def __init__(self, parent=None, *, max_blocks: int = 8000) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_blocks)
        self.setStyleSheet(
            "QPlainTextEdit{background:#1E1E1E; color:#D4D4D4;"
            "font-family:Consolas,'Courier New',monospace; font-size:11px;}")
        self._line.connect(self._append, QtCore.Qt.ConnectionType.QueuedConnection)

    @QtCore.Slot(str)
    def _append(self, msg: str) -> None:
        self.appendPlainText(msg.rstrip("\n"))
        self.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def log(self, msg: str) -> None:
        self._line.emit(str(msg))


# ─────────────────────────────────────────────────────────────────────────────
# ADF viewer (pyqtgraph)
# ─────────────────────────────────────────────────────────────────────────────

class AdfView(QtWidgets.QWidget):
    """ADF / virtual-image viewer with pan / zoom / level controls."""

    popoutRequested = QtCore.Signal()      # "⤢ Open" → host opens the overlay in a window

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        import numpy as np
        self._np = np
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        top = QtWidgets.QHBoxLayout()
        self._title = QtWidgets.QLabel("ADF — no image")
        self._title.setStyleSheet("font-size:9px; color:#1565C0;")
        top.addWidget(self._title, 1)
        self._chk_lines = QtWidgets.QCheckBox("Line profiles")
        self._chk_lines.setChecked(True)
        self._chk_lines.setStyleSheet("font-size:9px;")
        self._chk_lines.setToolTip(
            "Overlay line profiles (L1, L2, …) with color and label.\n"
            "Independent of the calibration ROI box.")
        self._chk_lines.toggled.connect(lambda _on: self._redraw_segs())
        top.addWidget(self._chk_lines)
        self._chk_roi = QtWidgets.QCheckBox("Cal. ROI")
        self._chk_roi.setChecked(True)
        self._chk_roi.setStyleSheet("font-size:9px;")
        self._chk_roi.setToolTip(
            "Overlay the calibration / strain reference ROI (cyan dashed box).\n"
            "Set in the ROI step; shared by ellipse, Q-pixel, and strain (with ROI).")
        self._chk_roi.toggled.connect(lambda _on: self._redraw_segs())
        top.addWidget(self._chk_roi)
        b_pop = QtWidgets.QToolButton(); b_pop.setText("⤢ Open")
        b_pop.setToolTip("Open this ADF + its lines (colored + labeled) in a separate window")
        b_pop.clicked.connect(lambda: self.popoutRequested.emit())
        top.addWidget(b_pop)
        lay.addLayout(top)

        import pyqtgraph as pg
        pg.setConfigOptions(imageAxisOrder="row-major")   # (row, col) like imshow
        self._pg = pg
        self._view = pg.ImageView()
        try:                                              # hide play/normalize chrome
            self._view.ui.roiBtn.hide()
            self._view.ui.menuBtn.hide()
        except Exception:
            pass
        lay.addWidget(self._view, 1)
        self._hlines: list = []
        self._segs: list = []          # live pg items
        self._seg_data: list = []      # last segments (so the toggle can redraw)
        self._roi_item = None          # live area-ROI rectangle item (legacy single)
        self._roi_data: list = []      # last area ROI [x0,x1,y0,y1] (legacy single)
        self._roi_items: list = []     # live multi-ROI items (rects + labels)
        self._rois_data: list = []     # last multi ROIs [(roi_id, [x0,x1,y0,y1]), …]

    def set_line_segments(self, segments) -> None:
        """Overlay line segments — accepts a DICT {line_id: [[x0,y0],[x1,y1]]} (drawn
        per-line COLORED + LABELED, matching the report) or a plain list of segments
        (legacy, single color). Shown only when 'Line profiles' is checked; [] clears."""
        if isinstance(segments, dict):
            self._seg_data = [(str(lid), seg) for lid, seg in segments.items()]
        else:
            self._seg_data = [(None, seg) for seg in (segments or [])]
        self._redraw_segs()

    def set_area_roi(self, bounds) -> None:
        """Overlay the analysis/area ROI [x0,x1,y0,y1] as a cyan rectangle (shown
        when 'Cal. ROI' is checked); pass [] to clear."""
        self._roi_data = list(bounds or [])
        self._redraw_segs()

    def set_area_rois(self, rois) -> None:
        """Overlay MULTIPLE analysis ROIs — accepts a DICT {roi_id: [x0,x1,y0,y1]}
        (drawn per-ROI COLORED + LABELED, matching the report). Shown when 'Cal. ROI'
        is checked; pass {} / [] to clear. Supersedes the single ``set_area_roi``."""
        if isinstance(rois, dict):
            self._rois_data = [(str(rid), list(b)) for rid, b in rois.items()
                               if b and len(b) == 4]
        else:
            self._rois_data = [(None, list(b)) for b in (rois or []) if b and len(b) == 4]
        self._redraw_segs()

    def _redraw_segs(self) -> None:
        view = self._view.getView()
        for it in self._segs:
            try:
                view.removeItem(it)
            except Exception:
                pass
        self._segs = []
        if self._roi_item is not None:
            try:
                view.removeItem(self._roi_item)
            except Exception:
                pass
            self._roi_item = None
        for it in self._roi_items:
            try:
                view.removeItem(it)
            except Exception:
                pass
        self._roi_items = []
        show_lines = self._chk_lines.isChecked()
        show_roi = self._chk_roi.isChecked()
        if not show_lines and not show_roi:
            return
        try:
            from engine import SIX_POINT_COLORS as _COLS
        except Exception:
            _COLS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                     "#46f0f0", "#f032e6", "#bcf60c"]
        if show_lines:
            for i, item in enumerate(self._seg_data):
                try:
                    lid, seg = item
                    (x0, y0), (x1, y1) = seg
                except Exception:
                    continue
                c = _COLS[i % len(_COLS)]              # per-line color (matches the report)
                it = self._pg.PlotDataItem([float(x0), float(x1)], [float(y0), float(y1)],
                                           pen=self._pg.mkPen(c, width=2.0))
                it.setZValue(10)
                view.addItem(it)
                self._segs.append(it)
                if lid:                                # per-line label at the start point
                    txt = self._pg.TextItem(str(lid), color=c, anchor=(0, 1))
                    txt.setPos(float(x0), float(y0))
                    txt.setZValue(11)
                    view.addItem(txt)
                    self._segs.append(txt)
        if show_roi and self._rois_data:                 # multi ROIs (colored + labeled)
            for i, (rid, b) in enumerate(self._rois_data):
                try:
                    x0, x1, y0, y1 = (float(v) for v in b)
                except Exception:
                    continue
                c = _COLS[i % len(_COLS)]
                it = self._pg.PlotDataItem([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                                           pen=self._pg.mkPen(c, width=1.7))
                it.setZValue(10)
                view.addItem(it)
                self._roi_items.append(it)
                if rid:
                    txt = self._pg.TextItem(str(rid), color=c, anchor=(0, 1))
                    txt.setPos(x0, y0)
                    txt.setZValue(11)
                    view.addItem(txt)
                    self._roi_items.append(txt)
        elif show_roi and len(self._roi_data) == 4:      # legacy single cyan rectangle
            x0, x1, y0, y1 = (float(v) for v in self._roi_data)
            it = self._pg.PlotDataItem([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                                       pen=self._pg.mkPen("#00E5FF", width=1.7))
            it.setZValue(10)
            view.addItem(it)
            self._roi_item = it

    def set_hlines(self, ys) -> None:
        """Overlay horizontal lines (line-tool row positions) on the ADF. Re-drawn
        each call: pass [] to clear. Lines are yellow InfiniteLines at each row y."""
        view = self._view.getView()
        for ln in self._hlines:
            try:
                view.removeItem(ln)
            except Exception:
                pass
        self._hlines = []
        for y in (ys or []):
            ln = self._pg.InfiniteLine(pos=float(y), angle=0,
                                       pen=self._pg.mkPen("#FFEB3B", width=1.5))
            ln.setZValue(10)
            view.addItem(ln)
            self._hlines.append(ln)

    def set_image(self, arr, *, title: str = "ADF") -> None:
        if arr is None:
            self._title.setText(f"{title} — no image")
            return
        a = self._np.asarray(arr, dtype=float)
        if a.ndim != 2:
            self._title.setText(f"{title} — not 2-D")
            return
        valid = a[a > 0]
        levels = (float(self._np.percentile(valid, 1)),
                  float(self._np.percentile(valid, 99))) if valid.size else None
        self._view.setImage(a, autoLevels=False,
                            levels=levels or (float(a.min()), float(a.max() or 1)))
        self._view.getView().invertY(True)               # origin at top, like imshow
        self._title.setText(f"{title}  ({a.shape[0]}×{a.shape[1]})")

    def clear(self) -> None:
        self._view.clear()
        self._title.setText("ADF — no image")


# ─────────────────────────────────────────────────────────────────────────────
# Probe viewer (the 2–4 images py4DSTEM produces when computing the probe)
# ─────────────────────────────────────────────────────────────────────────────

class ProbeView(QtWidgets.QWidget):
    """Shows the probe output — either a full matplotlib Figure (the 4-panel
    notebook view) via ``set_figure`` or a quick list of images via ``set_images``."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self._FC = FigureCanvasQTAgg
        self._Figure = Figure
        self._lay = QtWidgets.QVBoxLayout(self)
        self._lay.setContentsMargins(2, 2, 2, 2)
        self._canvas = None
        self.set_images([])

    def _show(self, fig) -> None:
        if self._canvas is not None:
            self._lay.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
        self._canvas = self._FC(fig)
        self._lay.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def set_figure(self, fig) -> None:
        """Display a ready-made matplotlib Figure (e.g. the 4-panel probe view)."""
        if fig is None:
            self.set_images([])
        else:
            self._show(fig)

    def set_images(self, images: list) -> None:
        """images: list of (title, 2D ndarray). Empty → placeholder text."""
        import numpy as np
        fig = self._Figure(figsize=(5, 3))
        if not images:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "No probe yet.\n"
                    "Set each scan's vacuum in Detection → Vacuum file,\n"
                    "then Compute probe (toolbar).",
                    ha="center", va="center", color="#888", fontsize=9)
            ax.axis("off")
        else:
            n = len(images)
            for i, (title, arr) in enumerate(images):
                ax = fig.add_subplot(1, n, i + 1)
                a = np.asarray(arr, dtype=float)
                v = a[a > 0]
                vmax = float(np.percentile(v, 99)) if v.size else (float(a.max()) or 1.0)
                ax.imshow(a, cmap="inferno", vmin=0, vmax=vmax, origin="upper")
                ax.set_title(title, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(pad=0.3)
        self._show(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration-figure thumbnails (per-file column in the param tabs) + maximize
# ─────────────────────────────────────────────────────────────────────────────

def figure_to_pixmap(fig, max_w: int = 240, max_h: int = 150, *, png_path: str = "", dpi: int = 80):
    """Render a matplotlib Figure (or a spilled PNG path) to a QPixmap thumbnail.

    Renders at the screen's device-pixel ratio (and, for live Figures, at a DPI
    high enough for the figure's own size) so the thumbnail stays sharp on HiDPI
    displays instead of being upscaled from a low-resolution raster."""
    screen = QtGui.QGuiApplication.primaryScreen()
    ratio = screen.devicePixelRatio() if screen else 1.0
    target_w, target_h = max_w * ratio, max_h * ratio
    if png_path:
        pix = QtGui.QPixmap(png_path)
        if not pix.isNull():
            pix = pix.scaled(int(target_w), int(target_h),
                             QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)
            pix.setDevicePixelRatio(ratio)
            return pix
        return QtGui.QPixmap()
    import io
    w_in, h_in = fig.get_size_inches()
    eff_dpi = min(max(dpi, target_w / w_in, target_h / h_in), 400)
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=eff_dpi, bbox_inches="tight")
    except Exception:
        return QtGui.QPixmap()
    pix = QtGui.QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    if not pix.isNull():
        pix = pix.scaled(int(target_w), int(target_h),
                         QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                         QtCore.Qt.TransformationMode.SmoothTransformation)
        pix.setDevicePixelRatio(ratio)
    return pix


_NAV_TB_CLS = None


def safe_nav_toolbar(canvas, parent):
    """A NavigationToolbar2QT whose ``set_message`` never crashes after its labels
    are destroyed. matplotlib keeps a mouse_move callback that, when the toolbar's
    QLabel is deleted (figure swapped / dialog closed), raises
    'libshiboken: Internal C++ object (QLabel) already deleted' on every mouse move
    and floods the console. Swallowing that one RuntimeError fixes it cleanly."""
    global _NAV_TB_CLS
    if _NAV_TB_CLS is None:
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

        class _SafeNavToolbar(NavigationToolbar2QT):
            def set_message(self, s):
                try:
                    super().set_message(s)
                except RuntimeError:
                    pass

        _NAV_TB_CLS = _SafeNavToolbar
    return _NAV_TB_CLS(canvas, parent)


class FigureDialog(QtWidgets.QDialog):
    """Maximized view of a matplotlib Figure with pan/zoom (NavigationToolbar)."""

    def __init__(self, fig, parent=None, title: str = "Figure") -> None:
        super().__init__(parent)
        _enable_minmax(self)
        self.setWindowTitle(title)
        self.resize(980, 760)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)  # prompt teardown
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        lay = QtWidgets.QVBoxLayout(self)
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(safe_nav_toolbar(canvas, self))
        lay.addWidget(canvas, 1)

    @classmethod
    def from_png(cls, path: str, parent=None, title: str = "Figure"):
        """Open a spilled PNG full-size (no matplotlib toolbar — static image)."""
        dlg = QtWidgets.QDialog(parent)
        _enable_minmax(dlg)
        dlg.setWindowTitle(title)
        dlg.resize(980, 760)
        lay = QtWidgets.QVBoxLayout(dlg)
        lab = QtWidgets.QLabel()
        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pix = QtGui.QPixmap(path)
        if not pix.isNull():
            lab.setPixmap(pix.scaled(940, 700, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                     QtCore.Qt.TransformationMode.SmoothTransformation))
        else:
            lab.setText("Could not load figure.")
        lay.addWidget(lab, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.accept)
        lay.addWidget(bb)
        return dlg


class ClickableFigureLabel(QtWidgets.QLabel):
    """A figure thumbnail; click → open it full-size (with residuals etc.).

    Pass ``scan``/``fig_key`` so the label re-resolves the Figure lazily on
    click via ``engine.resolve_figure`` (RAM-or-spilled-PNG) instead of holding
    its own permanent reference. A permanent reference can outlive
    ``FigurePolicy``'s eviction of the same Figure from ``scan.figures``
    (engine.py:1162-1178), keeping it — and the arrays it plots — resident
    longer than the policy intends. Omit ``scan``/``fig_key`` to keep today's
    behavior (used by qt_report.py's ad hoc report figures, which aren't part
    of the ``scan.figures`` pool in the first place).
    """

    def __init__(self, fig, *, spill_path: str = "", title: str = "Figure",
                 parent=None, max_w: int = 240, max_h: int = 150, dpi: int = 80,
                 scan=None, fig_key: str = "") -> None:
        super().__init__(parent)
        self._scan = scan
        self._fig_key = fig_key
        self._spill_path = spill_path or ""
        self._title = title
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(110, 84)
        if fig is not None:
            pix = figure_to_pixmap(fig, max_w, max_h, dpi=dpi)
            if not pix.isNull():
                self.setPixmap(pix)
            self.setToolTip("Click to enlarge")
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.setStyleSheet("border:1px solid #cfd8dc;")
        elif self._spill_path:
            pix = figure_to_pixmap(None, max_w, max_h, png_path=self._spill_path)
            if not pix.isNull():
                self.setPixmap(pix)
            self.setToolTip("Click to enlarge (spilled to disk)")
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.setStyleSheet("border:1px dashed #90A4AE;")
        else:
            self.setText("—")
            self.setStyleSheet("color:#aaa;")
        # Only cache a direct Figure reference when there's no (scan, fig_key) to
        # re-resolve from later — lazy mode is preferred whenever it's available.
        self._fig = fig if (scan is None or not fig_key) else None

    def _resolve_fig(self):
        if self._scan is not None and self._fig_key:
            import engine as E
            return E.resolve_figure(self._scan, self._fig_key)
        return self._fig

    def mousePressEvent(self, ev) -> None:
        fig = self._resolve_fig()
        if fig is not None:
            FigureDialog(fig, self.window(), self._title).exec()
        elif self._spill_path:
            FigureDialog.from_png(self._spill_path, self.window(), self._title).exec()


# ─────────────────────────────────────────────────────────────────────────────
# ADF gallery — thumbnail grid of every file's ADF (for multi-file)
# ─────────────────────────────────────────────────────────────────────────────

def adf_to_pixmap(arr, max_w: int = 150, max_h: int = 120):
    """Render a 2D ADF array to a grayscale QPixmap thumbnail (percentile levels)."""
    import numpy as np
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.size == 0:
        return QtGui.QPixmap()
    v = a[a > 0]
    lo, hi = (np.percentile(v, [1, 99]) if v.size else (float(a.min()), float(a.max())))
    if hi <= lo:
        hi = lo + 1.0
    g = np.ascontiguousarray((np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))
    h0, w0 = g.shape
    img = QtGui.QImage(g.data, w0, h0, w0, QtGui.QImage.Format.Format_Grayscale8)
    pix = QtGui.QPixmap.fromImage(img.copy())       # copy → own the buffer
    return pix.scaled(max_w, max_h, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                      QtCore.Qt.TransformationMode.SmoothTransformation)


def _line_overlay_colors():
    try:
        from engine import SIX_POINT_COLORS as cols
    except Exception:
        cols = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]
    return cols


def _draw_lines_on_pixmap(pm, lines, adf_shape, *, roi_bounds=None,
                          show_lines: bool = True, show_roi: bool = True):
    """Draw colored + labeled line segments (and optional ROI) on a thumbnail pixmap."""
    if pm.isNull() or not adf_shape:
        return pm
    h0, w0 = int(adf_shape[0]), int(adf_shape[1])
    sx = pm.width() / max(1, w0)
    sy = pm.height() / max(1, h0)
    has_lines = show_lines and bool(lines)
    has_roi = show_roi and roi_bounds and len(roi_bounds) == 4
    if not has_lines and not has_roi:
        return pm
    out = QtGui.QPixmap(pm)
    p = QtGui.QPainter(out)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    cols = _line_overlay_colors()
    if has_lines:
        items = (sorted(lines.items()) if isinstance(lines, dict)
                 else [(None, seg) for seg in (lines or [])])
        for i, (lid, seg) in enumerate(items):
            try:
                (x0, y0), (x1, y1) = seg
            except Exception:
                continue
            c = QtGui.QColor(cols[i % len(cols)])
            p.setPen(QtGui.QPen(c, max(1.0, 1.4)))
            p.drawLine(int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
            if lid:
                p.setPen(c)
                f = p.font()
                try:
                    fs = int(7 * min(float(sx), float(sy)))
                    fs = max(6, min(9, fs if fs > 0 else 7))
                except Exception:
                    fs = 7
                f.setPointSize(fs)
                p.setFont(f)
                p.drawText(int(x0 * sx) + 2, int(y0 * sy) - 2, str(lid))
    if has_roi:
        rx0, rx1, ry0, ry1 = (float(v) for v in roi_bounds)
        pen = QtGui.QPen(QtGui.QColor("#00E5FF"), max(1.0, 1.5))
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        p.setPen(pen)
        x0, x1 = int(rx0 * sx), int(rx1 * sx)
        y0, y1 = int(ry0 * sy), int(ry1 * sy)
        p.drawRect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
    p.end()
    return out


def _draw_segments_on_pixmap(pm, segs, adf_shape, *, roi_bounds=None):
    """Legacy wrapper — *segs* may be a ``lines`` dict or a plain segment list."""
    return _draw_lines_on_pixmap(pm, segs or {}, adf_shape, roi_bounds=roi_bounds)


class _GalleryCell(QtWidgets.QFrame):
    clicked = QtCore.Signal(int)

    def __init__(self, idx: int, name: str, arr, active: bool, parent=None,
                 *, overlay=None, show_lines: bool = True, show_roi: bool = True,
                 thumb_px: int = 150) -> None:
        super().__init__(parent)
        self._idx = idx
        overlay = overlay or {}
        lines = overlay.get("lines") if isinstance(overlay, dict) else overlay
        roi = overlay.get("roi", []) if isinstance(overlay, dict) else []
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QFrame{border:2px solid %s; border-radius:4px; background:#FAFAFA;}"
            % ("#1565C0" if active else "#E0E0E0"))
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(3, 3, 3, 3)
        v.setSpacing(1)
        tp = int(thumb_px)
        thumb = QtWidgets.QLabel()
        thumb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        thumb.setMinimumSize(tp, tp)
        if arr is not None:
            import numpy as np
            pm = adf_to_pixmap(arr, max_w=tp, max_h=tp)
            if (show_lines and lines) or (show_roi and roi):
                pm = _draw_lines_on_pixmap(pm, lines or {}, np.asarray(arr).shape,
                                           roi_bounds=roi, show_lines=show_lines,
                                           show_roi=show_roi)
            if not pm.isNull():
                thumb.setPixmap(pm)
        else:
            thumb.setText("not loaded")
            thumb.setStyleSheet("color:#aaa; border:none;")
        cap = QtWidgets.QLabel(name[:20])
        cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet("border:none; font-size:8px; color:%s;"
                          % ("#0D47A1" if active else "#555"))
        v.addWidget(thumb)
        v.addWidget(cap)

    def mousePressEvent(self, ev) -> None:
        self.clicked.emit(self._idx)


class AdfGallery(QtWidgets.QWidget):
    """Scrollable grid of per-file ADF thumbnails. Click a cell → ``selected(idx)``.

    Respects load-on-demand: files whose ADF isn't in memory show "not loaded";
    the "Load all ADFs" button asks the host to load them (``loadAllRequested``).
    """

    selected = QtCore.Signal(int)
    loadAllRequested = QtCore.Signal()

    def __init__(self, parent=None, *, cols: int = 3) -> None:
        super().__init__(parent)
        self._cols = cols
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        top = QtWidgets.QHBoxLayout()
        self._info = QtWidgets.QLabel("")
        self._info.setStyleSheet("color:#1565C0; font-size:9px;")
        self._chk_lines = QtWidgets.QCheckBox("Line profiles")
        self._chk_lines.setChecked(True)
        self._chk_lines.setStyleSheet("font-size:9px;")
        self._chk_lines.setToolTip(
            "Line profiles (L1, L2, …) with color and label on each thumbnail.")
        self._chk_lines.toggled.connect(lambda _on: self._redraw())
        self._chk_roi = QtWidgets.QCheckBox("Cal. ROI")
        self._chk_roi.setChecked(True)
        self._chk_roi.setStyleSheet("font-size:9px;")
        self._chk_roi.setToolTip(
            "Calibration / strain reference ROI (cyan dashed box) on each thumbnail.")
        self._chk_roi.toggled.connect(lambda _on: self._redraw())
        b_minus = QtWidgets.QPushButton("−")
        b_minus.setFixedWidth(26); b_minus.setToolTip("Smaller thumbnails (Ctrl+wheel)")
        b_minus.clicked.connect(lambda: self._set_thumb(self._thumb / 1.2))
        b_plus = QtWidgets.QPushButton("+")
        b_plus.setFixedWidth(26); b_plus.setToolTip("Bigger thumbnails (Ctrl+wheel)")
        b_plus.clicked.connect(lambda: self._set_thumb(self._thumb * 1.2))
        btn = QtWidgets.QPushButton("Load all ADFs")
        btn.clicked.connect(self.loadAllRequested)
        top.addWidget(self._info, 1)
        top.addWidget(self._chk_lines)
        top.addWidget(self._chk_roi)
        top.addWidget(b_minus)
        top.addWidget(b_plus)
        top.addWidget(btn)
        lay.addLayout(top)
        self._thumb: float = 150.0          # thumbnail size (px) — ± buttons / Ctrl+wheel
        self._last_items: list = []
        self._last_active: int = -1
        self._last_overlays: dict = {}
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._inner = QtWidgets.QWidget()
        self._grid = QtWidgets.QGridLayout(self._inner)
        self._grid.setSpacing(4)
        self._scroll.setWidget(self._inner)
        self._scroll.viewport().installEventFilter(self)   # Ctrl+wheel → resize
        lay.addWidget(self._scroll, 1)

    def _set_thumb(self, px: float) -> None:
        px = max(60.0, min(560.0, float(px)))
        if abs(px - self._thumb) > 0.5:
            self._thumb = px
            self._redraw()

    def eventFilter(self, obj, ev) -> bool:
        if (ev.type() == QtCore.QEvent.Type.Wheel
                and ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            self._set_thumb(self._thumb * (1.15 if ev.angleDelta().y() > 0 else 1 / 1.15))
            return True                                   # consume → no scroll
        return super().eventFilter(obj, ev)

    def refresh(self, items: list, active: int = -1, overlays_by_idx: dict | None = None) -> None:
        """items: list of (name, adf_ndarray_or_None).

        overlays_by_idx: {i: {"lines": {line_id: seg}, "roi": [x0,x1,y0,y1]}}}
        """
        self._last_items = list(items)
        self._last_active = active
        self._last_overlays = dict(overlays_by_idx or {})
        self._redraw()

    def _redraw(self) -> None:
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        items, active = self._last_items, self._last_active
        show_lines = self._chk_lines.isChecked()
        show_roi = self._chk_roi.isChecked()
        loaded = sum(1 for _n, a in items if a is not None)
        self._info.setText(f"{len(items)} file(s) · {loaded} ADF in memory")
        for i, (name, arr) in enumerate(items):
            cell = _GalleryCell(i, name, arr, i == active,
                                overlay=self._last_overlays.get(i),
                                show_lines=show_lines, show_roi=show_roi,
                                thumb_px=int(self._thumb))
            cell.clicked.connect(self.selected)
            self._grid.addWidget(cell, i // self._cols, i % self._cols)


# ─────────────────────────────────────────────────────────────────────────────
# Files tree — scan list with an expandable, explorable h5 root per file
# ─────────────────────────────────────────────────────────────────────────────

class FilesTree(QtWidgets.QTreeWidget):
    """Scan list where each file with an associated ``.h5`` is expandable to
    browse its root (groups / datasets). Double-clicking an array node asks the
    host to open it (``nodeActivated``); selecting any row picks the scan
    (``scanSelected``). The h5 root is populated lazily on expand (cheap: it reads
    structure only, never the heavy 4D payload), via the injected ``explore_fn``.
    """

    scanSelected = QtCore.Signal(int)         # top-level scan row (or its child) chosen
    nodeActivated = QtCore.Signal(int, str)   # (scan_idx, h5path-within-file) double-clicked

    _ROLE = QtCore.Qt.ItemDataRole.UserRole

    def __init__(self, parent=None, *, explore_fn=None) -> None:
        super().__init__(parent)
        self._explore = explore_fn            # callable(h5_file_path) -> nested dict | None
        self.setColumnCount(2)
        self.setHeaderLabels(["File / h5 root", "Info"])
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(True)
        hdr = self.header()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.currentItemChanged.connect(self._on_current)
        self.itemExpanded.connect(self._on_expanded)
        self.itemDoubleClicked.connect(self._on_double)

    # ── populate ───────────────────────────────────────────────────────────────
    def refresh(self, rows: list, active: int = -1) -> None:
        """rows: list of {label, h5path, tags}. One top-level item per scan; the h5
        root (if any) is a placeholder child expanded lazily. ``tags`` is a list of
        (name, present_bool) shown as check-icons in the Info column."""
        self.blockSignals(True)
        self.clear()
        for i, r in enumerate(rows):
            tags = r.get("tags", [])
            info = "  ".join(f"{nm} {'✓' if ok else '✗'}" for nm, ok in tags)
            it = QtWidgets.QTreeWidgetItem([r.get("label", ""), info])
            it.setData(0, self._ROLE, ("scan", i, r.get("h5path", "")))
            it.setToolTip(1, "  ".join(
                f"{nm}: {'present' if ok else 'missing'}" for nm, ok in tags))
            self.addTopLevelItem(it)
            if r.get("h5path"):
                ph = QtWidgets.QTreeWidgetItem(["(expand to explore h5 root)", ""])
                ph.setForeground(0, QtGui.QBrush(QtGui.QColor("#9e9e9e")))
                ph.setData(0, self._ROLE, ("placeholder", i, r["h5path"]))
                it.addChild(ph)
        if 0 <= active < self.topLevelItemCount():
            self.setCurrentItem(self.topLevelItem(active))
        self.blockSignals(False)

    def select_scan(self, idx: int) -> None:
        if 0 <= idx < self.topLevelItemCount():
            self.setCurrentItem(self.topLevelItem(idx))

    # ── lazy h5-root expansion ──────────────────────────────────────────────────
    def _on_expanded(self, item: QtWidgets.QTreeWidgetItem) -> None:
        data = item.data(0, self._ROLE)
        if not data or data[0] != "scan" or item.childCount() != 1:
            return
        child = item.child(0)
        cd = child.data(0, self._ROLE)
        if not cd or cd[0] != "placeholder":
            return
        item.removeChild(child)
        tree = self._explore(data[2]) if self._explore else None
        if not tree:
            err = QtWidgets.QTreeWidgetItem(["(could not read h5 root)", ""])
            err.setForeground(0, QtGui.QBrush(QtGui.QColor("#c62828")))
            item.addChild(err)
            return
        self._add_children(item, int(data[1]), tree)

    def _add_children(self, parent_item, scan_idx: int, node: dict) -> None:
        for ch in node.get("children", []):
            if ch["kind"] == "dataset":
                info = f"{ch['dtype']}  {tuple(ch['shape'])}"
            else:
                info = "group"
            twi = QtWidgets.QTreeWidgetItem([ch["name"], info])
            twi.setData(0, self._ROLE, ("node", scan_idx, ch["h5path"], ch["kind"]))
            if ch["kind"] == "dataset" and len(ch["shape"]) == 2:
                twi.setForeground(0, QtGui.QBrush(QtGui.QColor("#1565C0")))
                twi.setToolTip(0, "Double-click to view this image")
            parent_item.addChild(twi)
            if ch["kind"] == "group":
                self._add_children(twi, scan_idx, ch)

    # ── selection / activation ───────────────────────────────────────────────────
    def _on_current(self, cur, _prev) -> None:
        if cur is None:
            return
        data = cur.data(0, self._ROLE)
        if data and data[0] in ("scan", "node", "placeholder"):
            self.scanSelected.emit(int(data[1]))

    def _on_double(self, item, _col) -> None:
        data = item.data(0, self._ROLE)
        if data and data[0] == "node":
            self.nodeActivated.emit(int(data[1]), str(data[2]))


# ─────────────────────────────────────────────────────────────────────────────
# Interactive parameter tuner — a live preview + a column of knobs (debounced)
# ─────────────────────────────────────────────────────────────────────────────

class _LabeledSlider(QtWidgets.QWidget):
    """A horizontal slider with an editable value readout; supports int and float (via step)."""

    valueChanged = QtCore.Signal()

    def __init__(self, lo, hi, step, value, *, decimals: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._lo, self._hi = float(lo), float(hi)
        self._step, self._decimals = float(step), int(decimals)
        n = max(1, int(round((float(hi) - float(lo)) / float(step))))
        self._sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sl.setRange(0, n)
        self._val = QtWidgets.QLineEdit()
        self._val.setMinimumWidth(72)
        self._val.setMaximumWidth(96)
        self._val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                               | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if self._decimals:
            self._val.setValidator(QtGui.QDoubleValidator(self._lo, self._hi, self._decimals, self._val))
        else:
            self._val.setValidator(QtGui.QIntValidator(int(self._lo), int(self._hi), self._val))
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._sl, 1)
        lay.addWidget(self._val)
        self.setValue(value)
        self._sl.valueChanged.connect(self._on)
        self._val.editingFinished.connect(self._on_text)

    def _fmt(self, v) -> str:
        return f"{v:.{self._decimals}f}" if self._decimals else str(int(v))

    def _on(self) -> None:
        self._val.setText(self._fmt(self.value()))
        self.valueChanged.emit()

    def _on_text(self) -> None:
        try:
            v = float(self._val.text())
        except ValueError:
            v = self.value()
        v = max(self._lo, min(self._hi, v))
        self.setValue(v)
        self.valueChanged.emit()

    def value(self):
        v = self._lo + self._sl.value() * self._step
        return round(v, self._decimals) if self._decimals else int(round(v))

    def setValue(self, v) -> None:
        self._sl.blockSignals(True)
        self._sl.setValue(int(round((float(v) - self._lo) / self._step)))
        self._sl.blockSignals(False)
        self._val.setText(self._fmt(self.value()))

    def setMaximum(self, hi) -> None:
        """Widen/narrow the upper bound at runtime (e.g. once the probe size is known)."""
        self._hi = float(hi)
        cur = self.value()
        n = max(1, int(round((self._hi - self._lo) / self._step)))
        self._sl.blockSignals(True)
        self._sl.setMaximum(n)
        self._sl.setValue(int(round((float(cur) - self._lo) / self._step)))
        self._sl.blockSignals(False)
        if self._decimals:
            self._val.validator().setRange(self._lo, self._hi, self._decimals)
        else:
            self._val.validator().setRange(int(self._lo), int(self._hi))
        self._val.setText(self._fmt(self.value()))


class TunerDialog(QtWidgets.QDialog):
    """Generic live tuner: a matplotlib preview on the left, a column of knobs on
    the right. Reusable for detection (6-point grid) and basis (QR preview).

    Knobs edit attributes of ``obj`` (e.g. a CalibrationParams). View presets edit a
    local ``view`` dict (display-only dropdowns). Any change schedules a debounced
    recompute that calls ``render_fig(obj, view) -> matplotlib Figure`` OFF the GUI
    thread (so GPU detection doesn't freeze the UI) and swaps the canvas. ``Update``
    forces an immediate recompute; closing calls ``on_commit(obj)``.

    knob_specs : list of dicts — {attr, label, kind('int'|'float'|'enum'|'bool'),
                 min, max, step, decimals, values([(label,val)…] or [val…])}
    view_specs : list of dicts — {key, label, values([(label,val)…] or [val…])}
    """

    _sig_fig = QtCore.Signal(object)
    _sig_err = QtCore.Signal(str)

    def __init__(self, parent=None, *, title: str, obj, knob_specs: list,
                 view_specs: list | None = None, render_fig=None, view=None,
                 on_commit=None, debounce_ms: int = 300, extra_actions=None) -> None:
        super().__init__(parent)
        _enable_minmax(self)
        self._extra_actions = list(extra_actions or [])   # [(label, callable), …]
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self._FC = FigureCanvasQTAgg
        self.setWindowTitle(title)
        self.resize(1180, 760)
        self._obj = obj
        self._render = render_fig
        self._on_commit = on_commit
        self._view = dict(view or {})
        self._view_rows: dict = {}      # view key -> container QWidget (for depends_on)
        self._view_deps: dict = {}      # view key -> {"key": other_key, "in": [values]}
        self._busy = False
        self._pending = False
        self._canvas = None

        root = QtWidgets.QHBoxLayout(self)
        # left: preview
        left = QtWidgets.QVBoxLayout()
        self._canvas_host = QtWidgets.QVBoxLayout()
        cw = QtWidgets.QWidget(); cw.setLayout(self._canvas_host)
        left.addWidget(cw, 1)
        root.addLayout(left, 1)
        # right: knobs
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(360)
        form = QtWidgets.QVBoxLayout(panel)
        for _lbl, _cb in self._extra_actions:     # e.g. "Load data + probe" (top of the panel)
            _b = QtWidgets.QPushButton(_lbl); _b.clicked.connect(_cb)
            form.addWidget(_b)
        form.addWidget(self._h("Detection parameters"))
        for spec in knob_specs:
            form.addLayout(self._build_knob(spec))
        if view_specs:
            form.addWidget(self._h("Display (view filter — preset only)"))
            for spec in view_specs:
                form.addWidget(self._build_view(spec))
            self._update_view_visibility()
        form.addStretch(1)
        self._status = QtWidgets.QLabel("ready")
        self._status.setStyleSheet("color:#1565C0; font-size:10px;")
        self._err = QtWidgets.QLabel("")
        self._err.setWordWrap(True)
        self._err.setStyleSheet("color:#C62828; font-size:10px;")
        btns = QtWidgets.QHBoxLayout()
        b_up = QtWidgets.QPushButton("Update")
        b_up.clicked.connect(self._recompute)
        b_close = QtWidgets.QPushButton("Close")
        b_close.clicked.connect(self._on_close)
        btns.addWidget(self._status, 1); btns.addWidget(b_up); btns.addWidget(b_close)
        form.addWidget(self._err)
        form.addLayout(btns)
        root.addWidget(panel)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(debounce_ms))
        self._timer.timeout.connect(self._recompute)
        self._sig_fig.connect(self._on_fig)
        self._sig_err.connect(self._on_err)
        QtCore.QTimer.singleShot(0, self._recompute)        # initial render

    # ── widget builders ─────────────────────────────────────────────────────────
    def _h(self, text: str) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(text)
        lab.setStyleSheet("font-weight:bold; margin-top:6px;")
        return lab

    @staticmethod
    def _norm_values(values):
        out = []
        for v in values:
            out.append(v if isinstance(v, (tuple, list)) else (str(v), v))
        return out

    def _build_knob(self, spec: dict) -> QtWidgets.QVBoxLayout:
        box = QtWidgets.QVBoxLayout()
        box.setSpacing(1)
        attr, kind = spec["attr"], spec["kind"]
        cur = getattr(self._obj, attr)
        box.addWidget(QtWidgets.QLabel(spec["label"]))
        if kind in ("int", "float"):
            dec = spec.get("decimals", 0 if kind == "int" else 3)
            w = _LabeledSlider(spec["min"], spec["max"], spec["step"], cur, decimals=dec)
            w.valueChanged.connect(lambda a=attr, ww=w: self._set_attr(a, ww.value()))
            box.addWidget(w)
        elif kind == "enum":
            vals = self._norm_values(spec["values"])
            cb = QtWidgets.QComboBox()
            for lbl, val in vals:
                cb.addItem(lbl, val)
            idx = next((i for i, (_l, v) in enumerate(vals) if v == cur), 0)
            cb.setCurrentIndex(idx)
            cb.currentIndexChanged.connect(
                lambda _i, a=attr, c=cb: self._set_attr(a, c.currentData()))
            box.addWidget(cb)
        elif kind == "bool":
            chk = QtWidgets.QCheckBox("on")
            chk.setChecked(bool(cur))
            chk.toggled.connect(lambda v, a=attr: self._set_attr(a, bool(v)))
            box.addWidget(chk)
        return box

    def _build_view(self, spec: dict) -> QtWidgets.QWidget:
        box = QtWidgets.QVBoxLayout()
        box.setSpacing(1)
        box.setContentsMargins(0, 0, 0, 0)
        key = spec["key"]
        kind = spec.get("kind", "enum")
        box.addWidget(QtWidgets.QLabel(spec["label"]))
        if kind == "slider":
            dec = spec.get("decimals", 0)
            cur = self._view.get(key, spec.get("default", spec["min"]))
            self._view[key] = cur
            w = _LabeledSlider(spec["min"], spec["max"], spec["step"], cur, decimals=dec)
            w.valueChanged.connect(lambda k=key, ww=w: self._set_view(k, ww.value()))
            box.addWidget(w)
        else:
            vals = self._norm_values(spec["values"])
            cb = QtWidgets.QComboBox()
            for lbl, val in vals:
                cb.addItem(lbl, val)
            cur = self._view.get(key, vals[0][1])
            idx = next((i for i, (_l, v) in enumerate(vals) if v == cur), 0)
            cb.setCurrentIndex(idx)
            self._view[key] = cb.currentData()
            cb.currentIndexChanged.connect(lambda _i, k=key, c=cb: self._set_view(k, c.currentData()))
            box.addWidget(cb)
        container = QtWidgets.QWidget()
        container.setLayout(box)
        if "depends_on" in spec:
            self._view_deps[key] = spec["depends_on"]
        self._view_rows[key] = container
        return container

    # ── change → debounce ────────────────────────────────────────────────────────
    def _set_attr(self, attr, value) -> None:
        setattr(self._obj, attr, value)
        self._timer.start()

    def _set_view(self, key, value) -> None:
        self._view[key] = value
        if any(dep["key"] == key for dep in self._view_deps.values()):
            self._update_view_visibility()
        self._timer.start()

    def _update_view_visibility(self) -> None:
        for key, dep in self._view_deps.items():
            row = self._view_rows.get(key)
            if row is not None:
                row.setVisible(self._view.get(dep["key"]) in dep["in"])

    # ── recompute (threaded) ──────────────────────────────────────────────────────
    def _recompute(self) -> None:
        if self._render is None:
            return
        if self._busy:
            self._pending = True
            return
        self._busy = True
        self._err.setText("")
        self._status.setText("computing…")
        obj, view = self._obj, dict(self._view)

        def work():
            try:
                fig = self._render(obj, view)
                self._sig_fig.emit(fig)
            except Exception as exc:
                self._sig_err.emit(str(exc))

        threading.Thread(target=work, daemon=True, name="tuner").start()

    @QtCore.Slot(object)
    def _on_fig(self, fig) -> None:
        self._busy = False
        self._status.setText("ready")
        if fig is not None:
            self._show(fig)
        self._drain_pending()

    @QtCore.Slot(str)
    def _on_err(self, msg) -> None:
        self._busy = False
        self._status.setText("error")
        self._err.setText(msg)
        self._drain_pending()

    def _drain_pending(self) -> None:
        if self._pending:
            self._pending = False
            self._recompute()

    def _show(self, fig) -> None:
        if self._canvas is not None:
            self._canvas_host.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
        self._canvas = self._FC(fig)
        self._canvas_host.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def _on_close(self) -> None:
        if self._on_commit is not None:
            try:
                self._on_commit(self._obj)
            except Exception:
                pass
        self.accept()
