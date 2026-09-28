"""fast4d.qt_main — the Fast4D main window (PySide6, dockable).

A ``QMainWindow`` whose panels are all **movable / floatable / resizable**
``QDockWidget``s (the user's request), arranged around the Excel-like parameter
table:

    central : qt_params.ParamTable   (tabbed Excel grid, one tab per calib + ROI)
    left    : Files dock             (scan list + add raw/braggpeaks/template/ws)
    left    : Status dock            (ResourceMonitor + CalStateStrip)
    right   : ADF dock               (pyqtgraph viewer)
    bottom  : Console dock           (py4DSTEM messages)
    toolbars: icon strip (drives the tabs) + context step-actions (movable, wrap)
    bottom bar: strain options + Compute + progress

Heavy work runs off the GUI thread (``threading.Thread``); driver log/progress/
done callbacks are delivered back through Qt signals (thread-safe, queued).
``engine`` / ``driver`` / ``analysis`` are reused unchanged.
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from _bootstrap import bootstrap_sys_path, icon_path, ICONS_DIR

bootstrap_sys_path()
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import engine as E
import driver as D
from qt_params import ParamTable
from qt_widgets import (AdfGallery, AdfView, CalStateStrip, ConsoleWidget,
                        ExportReportDialog, ExportSelectionDialog, FilesTree, FlowLayout, ResourceMonitor,
                        _LabeledSlider)

_ICONS = ICONS_DIR


def _enable_minmax(dlg) -> None:
    """Give a dialog the standard title-bar buttons: minimize / maximize / close
    (Qt dialogs show only Close by default)."""
    dlg.setWindowFlags(dlg.windowFlags()
                       | QtCore.Qt.WindowType.WindowMinimizeButtonHint
                       | QtCore.Qt.WindowType.WindowMaximizeButtonHint)


def _is_visible(dlg) -> bool:
    try:
        return bool(dlg.isVisible())
    except Exception:
        return False


def _apply_channel_map_item(img, arr, scan, channel, *, pg_mod, cbar=None) -> tuple[float, float]:
    """Paint a live-tool ImageItem with the GUI colour range (not auto-percentiles)."""
    import numpy as np
    a = np.asarray(arr, dtype=float)
    lo, hi = E.channel_display_levels(scan, channel, a)
    img.setImage(a, autoLevels=False, levels=(lo, hi))
    cmap_name = E.channel_display_cmap(scan, channel)
    ch = str(channel).lower()
    if ch == "adf":
        img.setLookupTable(None)
    else:
        try:
            lut = pg_mod.colormap.get(cmap_name, source="matplotlib").getLookupTable(nPts=256)
            img.setLookupTable(lut)
        except Exception:
            pass
    if cbar is not None:
        try:
            cbar.setLevels((float(lo), float(hi)))
            cm = pg_mod.colormap.get(cmap_name, source="matplotlib")
            cbar.setColorMap(cm)
        except Exception:
            pass
    return float(lo), float(hi)


def _attach_live_colorbar(glw, pg_mod):
    """Standalone ColorBarItem beside a live-tool map (shows the GUI clim).

    Do not call ``setImageItem``: levels come from the GUI range via
    ``_apply_channel_map_item``, not from an interactive histogram.
    """
    try:
        cbar = pg_mod.ColorBarItem(width=16, interactive=False)
        glw.addItem(cbar, row=0, col=1)
        return cbar
    except Exception:
        return None


# icon strip: (step_key, label, icon stem). Mirrors the workflow order.
# Analysis / Live tools live in menu Tools — not a workflow step icon.
STEPS = [
    ("probe", "Probe", "probe"), ("select6", "6 Points", "select_6pts"),
    ("detection", "Detection", "detection"), ("roi", "ROI", "roi"),
    ("origin", "Origin", "origin"), ("ellipse", "Ellipse", "ellipse_cal"),
    ("qpixel", "Q Pixel", "q_pixel"), ("basis", "Basis", "basis"),
    ("strain", "Strain", "strain"),
]

# Workflow steps that only apply while building braggpeaks (Path B).
_PATH_B_STEPS = frozenset({"probe", "select6", "detection"})
_STEP_TOOLTIPS = {
    "probe": "Probe template — vacuum file, sample ROI, synthetic, or mean DP (Path B).",
    "select6": "Pick 6 ADF points for detection preview (Path B).",
    "detection": "Tune detection params and compute braggpeaks.h5 (Path B).",
    "roi": "Calibration ROI on the ADF.",
    "origin": "Origin center on the Bragg vector map.",
    "ellipse": "Elliptical distortion calibration.",
    "qpixel": "Q-pixel size calibration.",
    "basis": "Basis vectors / QR transform.",
    "strain": "Strain maps (Compute).",
}

_STATUS_ICON = {"pending": "○", "computed": "◐", "done": "✓", "error": "✗"}

# ── UI label standard (toolbar + calibration tool dialogs) ───────────────────
LBL_APPLY_CALIB = "Apply calibration"
LBL_SETTING_CALIB = "Setting {name} calibration"
_SETTING_BTN_STYLE = (
    "QPushButton{padding:5px 14px; margin:2px; border:1px solid #1565C0;"
    "border-radius:5px; background:#E3F2FD; color:#0D47A1; font-weight:600;}"
    "QPushButton:hover{background:#BBDEFB;}")
LBL_APPLY = "Apply"
LBL_THROUGH = "Through"
LBL_RESET = "Reset"
LBL_TO_TABLE = "To table"
LBL_FIT = "Fit"
LBL_PREP_UPSTREAM = "Prep upstream"
LBL_RERENDER_BVM = "Re-render BVM"

TIP_JUMP_OPEN = (
    "Open this step's interactive tool without running upstream calibrations.\n\n"
    "Use when the file is already calibrated through the previous step "
    "(e.g. Origin + Ellipse done — you only want to tweak Q-pixel).\n\n"
    "Does not walk tabs or apply upstream steps.")
TIP_JUMP_PREP = (
    "Guided jump: apply every calibration BEFORE this step on the active file "
    "(from the parameter table), then open the tool.\n\n"
    "Use to go straight to Q-pixel, Basis, etc. without visiting each tab.\n\n"
    "Same as Prep upstream in the tool dialog, then opening the tool.")
TIP_APPLY = (
    "Apply only this step on the active file (reads the parameter table). "
    "Restores this step's pre-step baseline first so re-applying never compounds.")
TIP_THROUGH = (
    "Run the full calibration chain through and including this step on the active file.")
TIP_RESET = (
    "Revert to the state before this step and clear downstream calibrations.")
TIP_PREP_UPSTREAM = (
    "Run calibrations before this step (Origin → …) on the selected file so it is "
    "ready for this tool. Only one file's braggpeaks state is in RAM at a time.")
TIP_FIT_ELLIPSE = (
    "Fit the ellipse on the chosen ring (r0, r1). Always starts from the pre-ellipse "
    "baseline — repeat Fit as many times as needed without compounding.")
TIP_COMMIT_ELLIPSE = (
    "Write the fitted ellipse to braggpeaks (restores pre-ellipse baseline before commit).")
TIP_TO_TABLE_ELLIPSE = (
    "Push q_range, sampling, Use ROI, and Enabled to the parameter table "
    "without applying the fit to braggpeaks — use before a second Fit/Apply cycle.")
TIP_TO_TABLE_QPIXEL = (
    "Push px guess, k_max, bragg_k_power, and Use ROI to the parameter table "
    "without running REFIT / Apply.")
TIP_TO_TABLE_BASIS = (
    "Push basis tuning parameters to the parameter table without applying "
    "to braggpeaks.")
PATH_A_DET_MSG = (
    "Path A — braggpeaks.h5 on disk. Skip probe/vacuum/datacube; go to ROI → Origin → …")
PATH_A_DET_TIP = (
    "This scan already has braggpeaks.h5. Calibration only needs a light braggpeaks load "
    "(auto on file select). Probe, vacuum, and Load data are Path B only.")

_CALIB_GUIDE_HTML = """
<h3>Calibration workflow — quick guide</h3>
<p><b>Three separate concerns (do not mix them up):</b></p>
<ul>
<li><b>ADF overlays</b> (Line profiles / Cal. ROI) — draw on top of the ADF view only.</li>
<li><b>Figures / RAM</b> — Report figures; close preview tools if you see many open figures.</li>
<li><b>Calibration</b> — how parameter-table values are applied to braggpeaks.</li>
</ul>
<hr>
<p><b>Just want lines or ROI on the ADF?</b><br>
→ Toggle <i>Line profiles</i> and/or <i>Cal. ROI</i> in the viewer or gallery.</p>
<p><b>Testing sliders and RAM fills up?</b><br>
→ Set <i>Figures</i> to <b>preview</b> or <b>off</b>; use <b>Clear figures</b> next to Free RAM.</p>
<p><b>Good params already — recompute strain only?</b><br>
→ <b>Compute</b> with Calib = <b>apply</b> (fast re-apply of known values).</p>
<p><b>New file or uncertain params?</b><br>
→ <b>Compute</b> with Calib = <b>fit</b> (recalibrate origin → ellipse → Q-pixel → basis).</p>
<p><b>Jump to Q-pixel / Ellipse without walking every tab?</b><br>
→ Step toolbar: <b>Setting … calibration</b> (checks status, applies upstream from the
table if needed, then opens the tool).</p>
<hr>
<p><b>Compute Calib — fit vs apply:</b></p>
<ul>
<li><b>fit</b> — re-runs origin, Q-pixel refit (when q_refit=ON), etc.</li>
<li><b>apply</b> — copies known table values; no refit.</li>
</ul>
<p><b>q_refit</b> (Q-pixel tab): ON = fit px from crystal; OFF = use table guess as px.</p>
<p><b>Ellipse re-fit:</b> each <i>Fit</i> starts from pre-ellipse state (no compounding).
<i>Commit</i> writes the result to braggpeaks.</p>
<p><b>Toolbar verbs:</b> <i>Apply</i> = this step only · <i>Through</i> = chain through this step ·
<i>Reset</i> = undo this step · <i>To table</i> = push tool values to the parameter table.</p>
<hr>
<p><b>Figures mode</b> (run bar):</p>
<ul>
<li><b>report</b> — keep calibration/strain figures in RAM + Report (Compute default).</li>
<li><b>preview</b> — tools show figures but do not register them (less RAM while tuning).</li>
<li><b>off</b> — discard figures immediately; best while exploring sliders.</li>
</ul>
<p><b>Max</b> — cap registered figures per scan; oldest preview slots are evicted first.</p>
<p><b>Spill</b> — when RAM cap evicts a figure, save a PNG sidecar (still visible in Report
with a dashed border; reloads on click).</p>
<p><b>Store…</b> — choose which figure types are kept when mode = report.</p>
<p><b>View DPI</b> — resolution for spilled / temp PNG sidecars (GUI thumbnails; lower = smaller files).<br>
<b>Save DPI</b> — resolution for <code>figures/</code> export on Compute / Save (raise for publication).</p>
"""

# Calibration steps: prep key for apply_calibrations_through, dialog opener.
_CALIB_META: dict[str, tuple[str | None, str]] = {
    "origin": (None, "_pick_origin"),
    "ellipse": ("ellipse", "_open_ellipse_tool"),
    "qpixel": ("qpixel", "_open_qpixel_tool"),
    "basis": ("basis", "_open_basis_tuner"),
    "roi": (None, "_open_roi_tool"),
}
_CALIB_TITLES = {
    "origin": "Origin", "ellipse": "Ellipse", "qpixel": "Q Pixel",
    "basis": "Basis", "roi": "ROI",
}


# ─────────────────────────────────────────────────────────────────────────────
# pyqtgraph picker (6 points / ROI / origin center) — Qt analogue of the Tk one
# ─────────────────────────────────────────────────────────────────────────────

class AdfPicker(QtWidgets.QDialog):
    """Pick on the ADF. mode: 'rect'→[x0,x1,y0,y1]; 'points'→[(x,y)…]; 'point'→[y,x]."""

    def __init__(self, parent, image, *, mode="rect", n_points=6, title="Pick"):
        super().__init__(parent)
        _enable_minmax(self)
        import numpy as np
        import pyqtgraph as pg
        self._np, self._pg = np, pg
        self._mode, self._n = mode, int(n_points)
        self._pts: list[tuple[float, float]] = []
        self.value = None
        self.setWindowTitle(title)
        self.resize(680, 720)

        lay = QtWidgets.QVBoxLayout(self)
        self._hint = QtWidgets.QLabel({
            "rect": "Drag the rectangle to set the ROI.",
            "points": f"Click {self._n} points (Clear to restart).",
            "point": "Click the diffraction-pattern center.",
            "hlines": "Click rows to add full-width horizontal lines. Clear to restart.",
            "lines": "Free: click 2 points per line.  Fix X: click a row.  Fix Y: click a column."}[mode])
        self._hl_items: list = []
        self._specs: list = []          # lines mode: typed specs (h/v/seg)
        self._pending = None            # free mode: first click of a 2-click line
        self._fix = "free"              # free | x | y
        lay.addWidget(self._hint)
        if mode == "lines":             # Free / Fix X / Fix Y selector
            radio = QtWidgets.QHBoxLayout()
            self._rb_free = QtWidgets.QRadioButton("Free (2 clicks)"); self._rb_free.setChecked(True)
            self._rb_x = QtWidgets.QRadioButton("Fix X (horizontal)")
            self._rb_y = QtWidgets.QRadioButton("Fix Y (vertical)")
            for rb, mk in ((self._rb_free, "free"), (self._rb_x, "x"), (self._rb_y, "y")):
                rb.toggled.connect(lambda on, m=mk: self._set_fix(m) if on else None)
                radio.addWidget(rb)
            radio.addStretch(1)
            lay.addLayout(radio)

        pg.setConfigOptions(imageAxisOrder="row-major")
        self._glw = pg.GraphicsLayoutWidget()
        self._vb = self._glw.addViewBox()
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)
        a = np.asarray(image, dtype=float)
        self._img = pg.ImageItem(a)
        v = a[a > 0]
        if v.size:
            self._img.setLevels([np.percentile(v, 1), np.percentile(v, 99)])
        self._vb.addItem(self._img)
        self._scatter = pg.ScatterPlotItem()
        self._vb.addItem(self._scatter)
        lay.addWidget(self._glw, 1)

        self._roi = None
        if mode == "rect":
            h, w = a.shape
            self._roi = pg.RectROI([w * 0.3, h * 0.3], [w * 0.4, h * 0.4],
                                   pen=pg.mkPen("lime", width=2))
            self._vb.addItem(self._roi)
        else:
            self._img.scene().sigMouseClicked.connect(self._on_click)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        b_clear = QtWidgets.QPushButton("Clear"); b_clear.clicked.connect(self._clear)
        b_cancel = QtWidgets.QPushButton("Cancel"); b_cancel.clicked.connect(self.reject)
        b_ok = QtWidgets.QPushButton("OK"); b_ok.clicked.connect(self._ok)
        for b in (b_clear, b_cancel, b_ok):
            btns.addWidget(b)
        lay.addLayout(btns)

    def _set_fix(self, m: str) -> None:
        self._fix = m
        self._pending = None

    def _redraw_lines(self) -> None:
        pg = self._pg
        for it in self._hl_items:
            try:
                self._vb.removeItem(it)
            except Exception:
                pass
        self._hl_items = []
        for spec in self._specs:
            t = spec.get("type")
            if t == "h":
                ln = pg.InfiniteLine(pos=float(spec["y"]), angle=0,
                                     pen=pg.mkPen("yellow", width=1.5))
            elif t == "v":
                ln = pg.InfiniteLine(pos=float(spec["x"]), angle=90,
                                     pen=pg.mkPen("cyan", width=1.5))
            else:
                p0, p1 = spec["p0"], spec["p1"]
                ln = pg.PlotDataItem([p0[0], p1[0]], [p0[1], p1[1]],
                                     pen=pg.mkPen("orange", width=1.8))
            self._vb.addItem(ln); self._hl_items.append(ln)
        self._scatter.setData([self._pending[0]] if self._pending else [],
                              [self._pending[1]] if self._pending else [],
                              size=10, pen=pg.mkPen("white"), brush=pg.mkBrush("red"))
        self._hint.setText(f"{len(self._specs)} line(s)"
                           + (" · click the 2nd point" if self._pending else ""))

    def _on_click(self, ev) -> None:
        p = self._vb.mapSceneToView(ev.scenePos())
        x, y = float(p.x()), float(p.y())
        if self._mode == "lines":
            if self._fix == "x":
                self._specs.append({"type": "h", "y": y})
            elif self._fix == "y":
                self._specs.append({"type": "v", "x": x})
            elif self._pending is None:
                self._pending = (x, y)
            else:
                self._specs.append({"type": "seg",
                                    "p0": [self._pending[0], self._pending[1]], "p1": [x, y]})
                self._pending = None
            self._redraw_lines()
            return
        if self._mode == "point":
            self._pts = [(x, y)]
        elif self._mode == "hlines":
            self._pts.append((x, y))                  # unlimited rows
        elif len(self._pts) < self._n:
            self._pts.append((x, y))
        self._redraw()

    def _redraw(self) -> None:
        if self._mode == "points":               # 6-point picker → colored markers (match detection)
            brushes = [self._pg.mkBrush(E.SIX_POINT_COLORS[i % len(E.SIX_POINT_COLORS)])
                       for i in range(len(self._pts))]
            self._scatter.setData([p[0] for p in self._pts], [p[1] for p in self._pts],
                                  size=15, pen=self._pg.mkPen("white", width=1.5), brush=brushes)
        else:
            self._scatter.setData([p[0] for p in self._pts], [p[1] for p in self._pts],
                                  size=12, pen=self._pg.mkPen("white"), brush=self._pg.mkBrush("red"))
        if self._mode == "hlines":
            for ln in self._hl_items:
                try:
                    self._vb.removeItem(ln)
                except Exception:
                    pass
            self._hl_items = []
            for (_x, y) in self._pts:
                ln = self._pg.InfiniteLine(pos=float(y), angle=0,
                                           pen=self._pg.mkPen("yellow", width=1.5))
                self._vb.addItem(ln); self._hl_items.append(ln)
            self._hint.setText(f"{len(self._pts)} line(s) — Clear to restart")
        if self._mode == "points":
            self._hint.setText(f"{len(self._pts)} / {self._n} points")

    def _clear(self) -> None:
        if self._mode == "lines":
            self._specs = []
            self._pending = None
            self._redraw_lines()
            return
        self._pts = []
        self._redraw()

    def _ok(self) -> None:
        if self._mode == "rect":
            x0, y0 = self._roi.pos(); w, h = self._roi.size()
            self.value = [round(x0), round(x0 + w), round(y0), round(y0 + h)]
        elif self._mode == "point":
            if not self._pts:
                self._hint.setText("Click a center point."); return
            x, y = self._pts[0]; self.value = [float(y), float(x)]
        elif self._mode == "hlines":
            if not self._pts:
                self._hint.setText("Click at least one row."); return
            self.value = sorted({int(round(y)) for (_x, y) in self._pts})
        elif self._mode == "lines":
            if not self._specs:
                self._hint.setText("Add at least one line."); return
            self.value = list(self._specs)
        else:
            if len(self._pts) != self._n:
                self._hint.setText(f"Need exactly {self._n} points."); return
            self.value = [(x, y) for (x, y) in self._pts]
        self.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Set up Lines — one window: ADF preview (with the placed lines) + the line tools
# ─────────────────────────────────────────────────────────────────────────────

class LineSetupDialog(QtWidgets.QDialog):
    """Set up the line profiles on the ADF: preview any file's ADF with its placed
    lines, and run the tools (pick rows, load lines from the loader JSON, load the
    drift CSV, propagate with/without drift). Delegates to the host window's
    line-tool methods so behaviour matches the Analysis-step buttons exactly."""

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        self._host = host
        self.setWindowTitle("Set up Lines")
        self.resize(940, 640)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from qt_widgets import safe_nav_toolbar
        self._FC = FigureCanvasQTAgg
        self._safe_tb = safe_nav_toolbar
        self._canvas = None
        self._tb = None

        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Preview file:"))
        self._combo = QtWidgets.QComboBox()
        for sc in host._scans:
            self._combo.addItem(sc.name)
        if 0 <= host._active < len(host._scans):
            self._combo.setCurrentIndex(host._active)
        self._combo.currentIndexChanged.connect(self._refresh)
        top.addWidget(self._combo, 1)
        left.addLayout(top)
        self._canvas_host = QtWidgets.QVBoxLayout()
        cw = QtWidgets.QWidget(); cw.setLayout(self._canvas_host)
        left.addWidget(cw, 1)
        lay.addLayout(left, 1)

        panel = QtWidgets.QWidget(); panel.setMaximumWidth(250)
        v = QtWidgets.QVBoxLayout(panel)
        # Apply to: lines/ROI actions (Pick, Load, Propagate, Clear) act only on
        # the checked files — e.g. propagate "with drift" to just the files that
        # matched a drift CSV row, leaving other (unrelated) files untouched.
        files_box = QtWidgets.QGroupBox("Apply to")
        fb = QtWidgets.QVBoxLayout(files_box)
        self._file_list = QtWidgets.QListWidget()
        self._file_list.setMaximumHeight(110)
        self._file_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        for sc in host._scans:
            it = QtWidgets.QListWidgetItem(self._file_item_label(sc))
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked)
            self._file_list.addItem(it)
        fb.addWidget(self._file_list)
        row1 = QtWidgets.QHBoxLayout()
        for txt, fn in (("All", self._check_all_files),
                        ("None", self._check_no_files),
                        ("Preview only", self._check_preview_only)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); row1.addWidget(b)
        fb.addLayout(row1)
        row2 = QtWidgets.QHBoxLayout()
        for txt, fn in (("With drift", self._check_with_drift),
                        ("Without drift", self._check_without_drift)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); row2.addWidget(b)
        fb.addLayout(row2)
        v.addWidget(files_box)
        v.addWidget(QtWidgets.QLabel("<b>Lines</b>  (same rows on every file)"))

        # Drift section — optional, for multi-file experiments with inter-scan drift
        drift_box = QtWidgets.QGroupBox("Inter-scan drift  (optional)")
        drift_box.setToolTip(
            "Use drift correction when the same sample region was scanned multiple times\n"
            "and the sample shifted slightly between acquisitions.\n\n"
            "Step 1: Estimate drift automatically from the loaded strain/ADF maps.\n"
            "Step 2: Or load a pre-computed drift CSV from a previous session.\n"
            "Step 3: Propagate with drift — each file gets its own shifted line position."
        )
        drift_lay = QtWidgets.QVBoxLayout(drift_box)
        drift_lay.setSpacing(3)
        for txt, fn in (
            ("Estimate drift…",   self._estimate_drift),
            ("Load drift CSV…",   self._load_drift),
        ):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); drift_lay.addWidget(b)
        v.addWidget(drift_box)

        for txt, fn in (("Pick rows on ADF…", self._pick),
                        ("Load lines from JSON…", self._load_json),
                        ("Propagate (no drift)", lambda: self._prop(False)),
                        ("Propagate (with drift)", lambda: self._prop(True)),
                        ("Clear lines", self._clear)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); v.addWidget(b)
        v.addSpacing(8)
        v.addWidget(QtWidgets.QLabel("<b>Area ROIs</b>  (multiple; same drift file as the lines)"))
        b_edit = QtWidgets.QPushButton("Edit ROIs…")
        b_edit.setStyleSheet(
            "QPushButton{background:#1565C0; color:white; font-weight:bold;"
            "border-radius:5px; padding:5px 14px;}"
            "QPushButton:hover{background:#1976D2;}")
        b_edit.setToolTip("Open the ROI editor: add, move, resize, duplicate and delete "
                          "all ROIs in one window — no OK per ROI.")
        b_edit.clicked.connect(self._edit_rois)
        v.addWidget(b_edit)
        for txt, fn in (("Load ROI from JSON…", self._load_roi),
                        ("Propagate ROIs (no drift)", lambda: self._prop_roi(False)),
                        ("Propagate ROIs (with drift)", lambda: self._prop_roi(True)),
                        ("Clear ROIs", self._clear_roi)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); v.addWidget(b)
        v.addStretch(1)
        self._status = QtWidgets.QLabel(""); self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:10px;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        v.addWidget(bb)
        lay.addWidget(panel)
        self._refresh()

    def _cur(self):
        i = self._combo.currentIndex()
        scans = self._host._scans
        return scans[i] if 0 <= i < len(scans) else None

    @staticmethod
    def _file_item_label(sc) -> str:
        dr = getattr(sc, "drift", None)
        tag = f"drift dx={dr[0]:+.1f} dy={dr[1]:+.1f}" if dr else "no drift"
        return f"{sc.name}  ·  {tag}"

    def _refresh(self, *_) -> None:
        sc = self._cur()
        if self._canvas is not None:
            self._canvas_host.removeWidget(self._canvas)
            self._canvas.setParent(None); self._canvas.deleteLater(); self._canvas = None
        if self._tb is not None:
            self._canvas_host.removeWidget(self._tb)
            self._tb.setParent(None); self._tb.deleteLater(); self._tb = None
        if sc is None:
            self._status.setText("No file selected."); return
        self._canvas = self._FC(E.build_lines_overlay_figure(sc))
        self._tb = self._safe_tb(self._canvas, self)
        self._canvas_host.addWidget(self._tb)
        self._canvas_host.addWidget(self._canvas, 1)
        self._canvas.draw_idle()
        n = len(sc.lines or {})
        roi = E.scan_display_roi(sc)
        self._status.setText(
            f"{sc.name}: {n} line(s)"
            + (f" + ROI {roi}" if len(roi) == 4 else " + no ROI")
            + (f"  ·  drift dx={sc.drift[0]:+.1f} dy={sc.drift[1]:+.1f}"
               if getattr(sc, 'drift', None) else "  ·  no drift"))
        scans = self._host._scans
        for i in range(min(self._file_list.count(), len(scans))):
            self._file_list.item(i).setText(self._file_item_label(scans[i]))

    # ── "Apply to" file checklist ───────────────────────────────────────────
    def _scope_targets(self) -> list:
        """Checked files in 'Apply to' (in scan order)."""
        scans = self._host._scans
        out = []
        for i in range(self._file_list.count()):
            if (self._file_list.item(i).checkState() == QtCore.Qt.CheckState.Checked
                    and i < len(scans)):
                out.append(scans[i])
        return out

    def _targets_or_warn(self):
        targets = self._scope_targets()
        if not targets:
            QtWidgets.QMessageBox.information(
                self, "Apply to", "Select at least one file in 'Apply to'.")
            return None
        return targets

    def _check_all_files(self) -> None:
        for i in range(self._file_list.count()):
            self._file_list.item(i).setCheckState(QtCore.Qt.CheckState.Checked)

    def _check_no_files(self) -> None:
        for i in range(self._file_list.count()):
            self._file_list.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)

    def _check_preview_only(self) -> None:
        ti = self._combo.currentIndex()
        for i in range(self._file_list.count()):
            st = QtCore.Qt.CheckState.Checked if i == ti else QtCore.Qt.CheckState.Unchecked
            self._file_list.item(i).setCheckState(st)

    def _check_with_drift(self) -> None:
        scans = self._host._scans
        for i in range(self._file_list.count()):
            has = i < len(scans) and bool(getattr(scans[i], "drift", None))
            self._file_list.item(i).setCheckState(
                QtCore.Qt.CheckState.Checked if has else QtCore.Qt.CheckState.Unchecked)

    def _check_without_drift(self) -> None:
        scans = self._host._scans
        for i in range(self._file_list.count()):
            has = i < len(scans) and bool(getattr(scans[i], "drift", None))
            self._file_list.item(i).setCheckState(
                QtCore.Qt.CheckState.Unchecked if has else QtCore.Qt.CheckState.Checked)

    # ── Lines ────────────────────────────────────────────────────────────────
    def _pick(self) -> None:
        targets = self._targets_or_warn()
        if targets is None:
            return
        self._host._template_idx = self._combo.currentIndex()
        self._host._pick_lines(targets=targets)
        self._refresh()

    def _load_json(self) -> None:
        targets = self._targets_or_warn()
        if targets is None:
            return
        self._host._load_lines_json(targets=targets)
        self._refresh()

    def _estimate_drift(self) -> None:
        self._host._show_tool(DriftEstimateDialog(self._host))

    def _load_drift(self) -> None:
        self._host._load_drift_csv()
        self._refresh()

    def _prop(self, drift: bool) -> None:
        targets = self._targets_or_warn()
        if targets is None:
            return
        self._host._propagate_lines(use_drift=drift, targets=targets)
        self._refresh()

    def _clear(self) -> None:
        targets = self._targets_or_warn()
        if targets is None:
            return
        self._host._template_lines = []
        for sc in targets:
            sc.lines = {}
        self._host._update_active_views()
        self._refresh()

    # ── Area ROIs ────────────────────────────────────────────────────────────
    def _edit_rois(self) -> None:
        """Open the single-window ROI editor (add / move / resize / duplicate /
        delete all ROIs with a live parameter table — no OK per ROI)."""
        self._host._active = self._combo.currentIndex()
        dlg = AreaRoiEditorDialog(self._host)
        dlg.exec()
        self._refresh()

    def _load_roi(self) -> None:
        targets = self._targets_or_warn()
        if targets is None:
            return
        self._host._load_roi_json(targets=targets)
        self._refresh()

    def _prop_roi(self, drift: bool) -> None:
        targets = self._targets_or_warn()
        if targets is None:
            return
        self._host._propagate_roi(use_drift=drift, targets=targets)
        self._refresh()

    def _clear_roi(self) -> None:
        targets = self._targets_or_warn()
        if targets is None:
            return
        self._host._template_roi = []
        for sc in targets:
            sc.area_roi = []
            sc.area_rois = {}
        self._host._update_active_views()
        self._refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Shared: a "File:" selector (loaded scans) + optional "Apply previous calibrations"
# for the calibration dialogs — so each works like the BF/ADF plugin (pick the file)
# and, for non-origin steps, you can run the prior calibrations on the picked file
# (only one file's state is in RAM at a time).
# ─────────────────────────────────────────────────────────────────────────────

def _add_calib_file_label(dlg, host, layout) -> None:
    """Show the active scan (Files panel selection) — one file at a time."""
    sc = host.active_scan()
    lbl = QtWidgets.QLabel(f"File: <b>{sc.name if sc else '(none selected)'}</b>")
    lbl.setWordWrap(True)
    layout.addWidget(lbl)
    dlg._sc = sc


# ─────────────────────────────────────────────────────────────────────────────
# Origin tool — pick the center on the BVM (starts at the probe center), live X/Y,
# sampling with explicit Apply. Handles the BVM sampling scale: measure_origin wants
# center_guess in DIFFRACTION px, but the BVM at sampling s shows features at s× —
# so a click at BVM (bx,by) → diffraction (bx/s, by/s). center_guess is stored (y,x).
# ─────────────────────────────────────────────────────────────────────────────

class OriginDialog(QtWidgets.QDialog):
    """Pick the diffraction center on the Bragg vector map. The red marker starts at
    the probe center (state.probe_qx0=row, probe_qy0=col) the first time; X (col) and
    Y (row) read out live (and are editable) so a swapped/inverted pick is obvious.
    The sampling slider only re-renders the BVM when you press Apply (not live)."""

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        import numpy as np
        import pyqtgraph as pg
        self._np, self._pg = np, pg
        self._host = host
        self._sc = host.active_scan()
        self._sampling = int((self._sc.params.origin_sampling if self._sc else 2) or 2)
        self._cx = 128.0    # DIFFRACTION-px column (x = qy)
        self._cy = 128.0    # DIFFRACTION-px row    (y = qx)
        self.setWindowTitle(f"Origin — pick center (BVM) — {self._sc.name if self._sc else ''}")
        self.resize(900, 740)

        lay = QtWidgets.QHBoxLayout(self)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._glw = pg.GraphicsLayoutWidget()
        self._vb = self._glw.addViewBox()
        self._vb.setAspectLocked(True); self._vb.invertY(True)
        self._img = pg.ImageItem(); self._vb.addItem(self._img)
        self._ring = pg.ScatterPlotItem(); self._ring.setZValue(10); self._vb.addItem(self._ring)
        self._img.scene().sigMouseClicked.connect(self._on_click)
        lay.addWidget(self._glw, 1)

        panel = QtWidgets.QWidget(); panel.setMaximumWidth(300)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel("<b>Origin</b> — click to set the center"))
        _add_calib_file_label(self, host, v)
        hint = QtWidgets.QLabel("X = column (qy) · Y = row (qx). Stored as center_guess (y, x).\n"
                                "If a pick lands swapped, the live X/Y will show it.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#1565C0; font-size:10px;")
        v.addWidget(hint)
        # live, editable X / Y (diffraction px)
        self._sp_x = self._dspin(v, "X = qy (col)", -1e4, 1e4, self._cx, 0.5, 2)
        self._sp_y = self._dspin(v, "Y = qx (row)", -1e4, 1e4, self._cy, 0.5, 2)
        self._sp_x.valueChanged.connect(self._on_spin)
        self._sp_y.valueChanged.connect(self._on_spin)
        # sampling slider + EXPLICIT Apply
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("sampling:"))
        self._sld_s = _LabeledSlider(1, 8, 1, self._sampling, decimals=0)
        srow.addWidget(self._sld_s, 1)
        v.addLayout(srow)
        b_apply_s = QtWidgets.QPushButton(LBL_RERENDER_BVM)
        b_apply_s.setToolTip("Re-render the Bragg vector map at the current sampling.")
        b_apply_s.clicked.connect(self._apply_sampling); v.addWidget(b_apply_s)
        b_probe = QtWidgets.QPushButton("Reset to probe")
        b_probe.clicked.connect(self._reset_to_probe)
        if E.analysis_path(self._sc) == "A" and self._probe_center() is None:
            b_probe.setEnabled(False)
            b_probe.setToolTip(
                "Path A — braggpeaks loaded; pick the center on the BVM or edit X/Y "
                "(probe/vacuum not required).")
        v.addWidget(b_probe)
        v.addStretch(1)
        self._status = QtWidgets.QLabel("Ready"); self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        v.addWidget(self._status)
        b_use = QtWidgets.QPushButton(LBL_TO_TABLE)
        b_use.setToolTip(
            "Push center_guess + sampling to the parameter table only.\n"
            "Does NOT write Origin onto braggpeaks yet.\n\n"
            "Next: step toolbar → Apply calibration (Origin), or Compute with Calib=fit.")
        b_use.clicked.connect(self._use_center); v.addWidget(b_use)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        lay.addWidget(panel)

        self._init_coords()
        self._reload_bvm()

    def _dspin(self, layout, label, lo, hi, val, step, dec):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(100)
        s = QtWidgets.QDoubleSpinBox(); s.setDecimals(int(dec)); s.setRange(float(lo), float(hi))
        s.setSingleStep(float(step)); s.setValue(float(val))
        row.addWidget(lbl); row.addWidget(s, 1); layout.addLayout(row)
        return s

    def _probe_center(self):
        """(row, col) probe center from state.probe_qx0/qy0, or None."""
        st = getattr(self._sc, "state", None)
        if st is None:
            return None
        pqx, pqy = getattr(st, "probe_qx0", None), getattr(st, "probe_qy0", None)
        if pqx is None or pqy is None:
            return None
        return (float(pqx), float(pqy))      # (row, col) = (y, x)

    def _init_coords(self):
        cg = (self._sc.params.center_guess if self._sc else None) or [128.0, 128.0]
        is_default = abs(float(cg[0]) - 128.0) < 1e-6 and abs(float(cg[1]) - 128.0) < 1e-6
        probe = self._probe_center()
        if probe is not None and is_default:     # first time → start at the probe center
            self._cy, self._cx = probe          # (row=y, col=x)
            self._status.setText(f"Started at probe center (qx0,qy0)=({probe[0]:.1f},{probe[1]:.1f}).")
        else:
            self._cy, self._cx = float(cg[0]), float(cg[1])   # center_guess = (y, x)

    def _reload_bvm(self):
        if self._sc is None:
            self._status.setText("No active scan."); return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            img = E.bragg_vector_map(self._sc, sampling=self._sampling, log=self._host._console.log)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._status.setText(f"BVM unavailable: {exc}"); return
        QtWidgets.QApplication.restoreOverrideCursor()
        if img is None:
            self._status.setText("No Bragg vector map."); return
        a = self._np.asarray(img, dtype=float)
        self._img.setImage(a)
        v = a[a > 0]
        if v.size:
            self._img.setLevels([float(self._np.percentile(v, 1)), float(self._np.percentile(v, 99))])
        self._redraw_marker()                    # keeps the current diffraction coords
        self._vb.autoRange()

    def _redraw_marker(self):
        s = float(self._sampling)
        bx, by = self._cx * s, self._cy * s      # diffraction → BVM-pixel coords
        self._ring.setData([bx], [by], size=18, symbol="o",
                           pen=self._pg.mkPen("red", width=2.5), brush=None)
        for sp, val in ((self._sp_x, self._cx), (self._sp_y, self._cy)):
            sp.blockSignals(True); sp.setValue(float(val)); sp.blockSignals(False)

    def _on_click(self, ev):
        p = self._vb.mapSceneToView(ev.scenePos())
        s = float(self._sampling)
        self._cx = float(p.x()) / s              # BVM-pixel → diffraction px
        self._cy = float(p.y()) / s
        self._redraw_marker()
        self._status.setText(f"center_guess (y,x) = ({self._cy:.2f}, {self._cx:.2f})  "
                             f"[diffraction px; sampling={self._sampling}]")

    def _on_spin(self, *_):
        self._cx = float(self._sp_x.value())
        self._cy = float(self._sp_y.value())
        self._redraw_marker()

    def _apply_sampling(self):
        self._sampling = int(self._sld_s.value())
        self._reload_bvm()                       # only re-renders on Apply
        self._status.setText(f"BVM re-rendered at sampling={self._sampling}.")

    def _reset_to_probe(self):
        probe = self._probe_center()
        if probe is None:
            self._status.setText("No probe center available (compute the probe first).")
            return
        self._cy, self._cx = probe
        self._redraw_marker()
        self._status.setText(f"Reset to probe center (qx0,qy0)=({probe[0]:.1f},{probe[1]:.1f}).")

    def _use_center(self):
        if self._sc is None:
            return
        self._sc.params.center_guess = [float(self._cy), float(self._cx)]   # (y, x)
        self._sc.params.origin_sampling = int(self._sampling)
        self._host._console.log(
            f"[{self._sc.name}] center_guess (y,x) = ({self._cy:.2f}, {self._cx:.2f}), "
            f"sampling={self._sampling}")
        self._host._console.log(
            f"[{self._sc.name}] NOTE: Origin is NOT on braggpeaks yet — "
            f"press Apply calibration on the Origin step (or Compute with Calib=fit). "
            f"Q-pixel fit needs calstate.center=True.")
        self._host._params.reload()
        self.accept()

    def _on_file_picked(self, sc):
        self._sc = sc
        self._sampling = int((sc.params.origin_sampling if sc else 2) or 2)
        self._sld_s.setValue(self._sampling)
        self._init_coords()
        self._reload_bvm()
        self.setWindowTitle(f"Origin — pick center (BVM) — {sc.name if sc else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# ROI tool — calibration ROI on the ADF: draggable rectangle + manual x0/x1/y0/y1,
# file selector, Apply to this file / ALL files. (ROI is the FIRST calib step.)
# ─────────────────────────────────────────────────────────────────────────────

class ROIDialog(QtWidgets.QDialog):
    """Set the calibration ROI [x0,x1,y0,y1] on the ADF: drag/resize the rectangle OR
    type the bounds; pick the file from the cascade; Apply to this file or ALL files
    (the ROI feeds ellipse/Q-pixel and the with-ROI strain reference)."""

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        import numpy as np
        import pyqtgraph as pg
        self._np, self._pg = np, pg
        self._host = host
        self._sc = host.active_scan()
        self._sync = False
        self.setWindowTitle(f"ROI — {self._sc.name if self._sc else ''}")
        self.resize(900, 680)
        lay = QtWidgets.QHBoxLayout(self)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._glw = pg.GraphicsLayoutWidget()
        self._vb = self._glw.addViewBox()
        self._vb.setAspectLocked(True); self._vb.invertY(True)
        self._img = pg.ImageItem(); self._vb.addItem(self._img)
        self._roi = pg.RectROI([10, 10], [40, 40], pen=pg.mkPen("lime", width=2))
        self._vb.addItem(self._roi)
        self._roi.sigRegionChanged.connect(self._roi_to_spins)
        lay.addWidget(self._glw, 1)

        panel = QtWidgets.QWidget(); panel.setMaximumWidth(280)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel("<b>ROI</b> — drag the rectangle or type bounds"))
        _add_calib_file_label(self, host, v)
        self._sp = {}
        for k in ("x0", "x1", "y0", "y1"):
            self._sp[k] = self._spin(v, k, 0, 100000)
            self._sp[k].valueChanged.connect(self._spins_to_roi)
        for txt, fn in (("Apply (file)", lambda: self._apply(False)),
                        ("Apply (all)", lambda: self._apply(True))):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); v.addWidget(b)
        v.addStretch(1)
        self._status = QtWidgets.QLabel("Ready"); self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept); v.addWidget(bb)
        lay.addWidget(panel)
        self._reload()

    def _spin(self, layout, label, lo, hi):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(30)
        s = QtWidgets.QSpinBox(); s.setRange(int(lo), int(hi))
        row.addWidget(lbl); row.addWidget(s, 1); layout.addLayout(row)
        return s

    def _reload(self):
        sc = self._sc
        adf = E.cached_adf(sc) if sc else None
        if adf is None:
            self._status.setText("No ADF for this file (load it first)."); return
        a = self._np.asarray(adf, dtype=float)
        self._img.setImage(a)
        v = a[a > 0]
        if v.size:
            self._img.setLevels([float(self._np.percentile(v, 1)), float(self._np.percentile(v, 99))])
        h, w = a.shape
        for k, hi in (("x0", w), ("x1", w), ("y0", h), ("y1", h)):
            self._sp[k].setMaximum(int(hi))
        b = list(sc.params.roi_bounds or []) if sc else []
        if len(b) != 4:
            b = [int(w * 0.3), int(w * 0.7), int(h * 0.3), int(h * 0.7)]
        self._set_bounds(b)
        self._vb.autoRange()
        self._status.setText(f"{sc.name}: ROI {self._bounds()}")

    def _bounds(self):
        x0, x1 = sorted((self._sp["x0"].value(), self._sp["x1"].value()))
        y0, y1 = sorted((self._sp["y0"].value(), self._sp["y1"].value()))
        return [x0, x1, y0, y1]

    def _set_bounds(self, b):
        self._sync = True
        x0, x1, y0, y1 = (int(v) for v in b)
        self._sp["x0"].setValue(x0); self._sp["x1"].setValue(x1)
        self._sp["y0"].setValue(y0); self._sp["y1"].setValue(y1)
        self._roi.setPos([x0, y0]); self._roi.setSize([max(1, x1 - x0), max(1, y1 - y0)])
        self._sync = False

    def _roi_to_spins(self, *_):
        if self._sync:
            return
        self._sync = True
        x0, y0 = self._roi.pos(); w, h = self._roi.size()
        self._sp["x0"].setValue(int(round(x0))); self._sp["x1"].setValue(int(round(x0 + w)))
        self._sp["y0"].setValue(int(round(y0))); self._sp["y1"].setValue(int(round(y0 + h)))
        self._sync = False
        self._status.setText(f"ROI {self._bounds()}")

    def _spins_to_roi(self, *_):
        if self._sync:
            return
        self._sync = True
        x0, x1, y0, y1 = self._bounds()
        self._roi.setPos([x0, y0]); self._roi.setSize([max(1, x1 - x0), max(1, y1 - y0)])
        self._sync = False
        self._status.setText(f"ROI {self._bounds()}")

    def _apply(self, all_files):
        if self._sc is None:
            return
        b = self._bounds()
        targets = self._host._scans if all_files else [self._sc]
        for s in targets:
            s.params.roi_bounds = list(b)
        self._host._params.reload()
        self._host._update_active_views()
        self._status.setText(f"ROI {b} applied to {len(targets)} file(s).")

    def _on_file_picked(self, sc):
        self._sc = sc
        self.setWindowTitle(f"ROI — {sc.name if sc else ''}")
        self._reload()


# ─────────────────────────────────────────────────────────────────────────────
# Ellipse calibration tool — pick the ring on the BVM, Fit, Apply (checkpoint-reset)
# ─────────────────────────────────────────────────────────────────────────────

class EllipseDialog(QtWidgets.QDialog):
    """Interactive ellipse calibration: shows the calibrated Bragg vector map, lets
    the user pick the ring (q_range r0,r1) with sliders or Ctrl-click, Fit the
    ellipse over that ring (no apply), and Apply it — Apply first resets to the
    pre-ellipse checkpoint so it never compounds on a previous ellipse."""

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        import numpy as np
        import pyqtgraph as pg
        self._host = host
        self._np, self._pg = np, pg
        self._sc = host.active_scan()
        self._fit = None
        self._cached_bvm = None
        self._bvm_pending = False
        self._origin = (0.0, 0.0)
        self._center_yx = None   # custom fit center (row,col); None until first BVM load
        self.setWindowTitle(f"Ellipse calibration — {self._sc.name if self._sc else ''}")
        self.resize(940, 700)

        lay = QtWidgets.QHBoxLayout(self)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._glw = pg.GraphicsLayoutWidget()
        self._vb = self._glw.addViewBox()
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)
        self._img = pg.ImageItem()
        self._vb.addItem(self._img)
        self._c0 = pg.PlotDataItem(pen=pg.mkPen("#FFEB3B", width=1.4))
        self._c1 = pg.PlotDataItem(pen=pg.mkPen("#FFEB3B", width=1.4))
        self._ell = pg.PlotDataItem(pen=pg.mkPen("#00E5FF", width=2.0))
        self._ctr = pg.ScatterPlotItem()
        self._ctr_custom = pg.ScatterPlotItem()
        for it in (self._c0, self._c1, self._ell, self._ctr, self._ctr_custom):
            it.setZValue(10); self._vb.addItem(it)
        self._img.scene().sigMouseClicked.connect(self._on_click)
        lay.addWidget(self._glw, 1)

        panel = QtWidgets.QWidget(); panel.setMaximumWidth(290)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel("<b>Ellipse</b> — pick the ring, then Fit"))
        _add_calib_file_label(self, host, v)
        hint = QtWidgets.QLabel("Ctrl + Left-click → r0   ·   Ctrl + Right-click → r1")
        hint.setStyleSheet("color:#1565C0; font-size:10px;")
        v.addWidget(hint)
        self._use_roi = QtWidgets.QCheckBox("Use ROI")
        self._use_roi.setChecked(bool(self._sc.params.ellipse_use_roi) if self._sc else False)
        v.addWidget(self._use_roi)
        self._s_samp = self._add_slider(v, "sampling", 1, 30,
                                        int(self._sc.params.ellipse_sampling or 1) if self._sc else 1)
        r0d, r1d = sorted(int(x) for x in
                          ((self._sc.params.ellipse_q_range if self._sc else None) or [30, 44]))
        self._s_r0 = self._add_slider(v, "q_range[0]", 0, 400, r0d)
        self._s_r1 = self._add_slider(v, "q_range[1]", 0, 400, r1d)
        for s in (self._s_r0, self._s_r1):
            s.valueChanged.connect(self._refresh_view)
        b_apply_bvm = QtWidgets.QPushButton(LBL_RERENDER_BVM)
        b_apply_bvm.setToolTip("Recompute the Bragg vector map (slow on large scans — "
                               "not live on every slider tick).")
        b_apply_bvm.clicked.connect(self._reload_bvm)
        v.addWidget(b_apply_bvm)

        self._chk_center = QtWidgets.QCheckBox(
            "Custom ellipse center (Shift+click to set)")
        self._chk_center.toggled.connect(self._on_center_toggled)
        v.addWidget(self._chk_center)
        cy0, cx0 = self._origin
        self._sp_cx = self._dspin(v, "center X (col)", -1e4, 1e4, cx0, 0.5, 2)
        self._sp_cy = self._dspin(v, "center Y (row)", -1e4, 1e4, cy0, 0.5, 2)
        self._sp_cx.valueChanged.connect(self._on_center_spin)
        self._sp_cy.valueChanged.connect(self._on_center_spin)
        b_center_reset = QtWidgets.QPushButton("Use BVM origin")
        b_center_reset.setToolTip("Reset the custom fit center to bvm.origin "
                                   "(the calibrated probe origin).")
        b_center_reset.clicked.connect(self._reset_center_to_origin)
        v.addWidget(b_center_reset)

        for txt, fn, tip in ((LBL_FIT, self._fit_ellipse, TIP_FIT_ELLIPSE),
                               (LBL_TO_TABLE, self._to_params, TIP_TO_TABLE_ELLIPSE),
                               (LBL_APPLY_CALIB, self._apply, TIP_COMMIT_ELLIPSE),
                               (LBL_RESET, self._reset, TIP_RESET)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn)
            if tip:
                b.setToolTip(tip)
            v.addWidget(b)
            if txt == LBL_APPLY_CALIB:
                self._btn_apply = b; b.setEnabled(False)
        v.addStretch(1)
        self._status = QtWidgets.QLabel("Ready"); self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        v.addWidget(bb)
        lay.addWidget(panel)
        self._reload_bvm()

    def _add_slider(self, layout, label, lo, hi, val):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(f"{label}:")
        lbl.setMinimumWidth(110)
        s = _LabeledSlider(lo, hi, 1, val, decimals=0)
        row.addWidget(lbl); row.addWidget(s, 1)
        layout.addLayout(row)
        return s

    def _circle(self, cx, cy, r, n=180):
        t = self._np.linspace(0, 2 * self._np.pi, n)
        return cx + r * self._np.cos(t), cy + r * self._np.sin(t)

    def _ellipse_pts(self, cx, cy, a, b, theta, n=180):
        np = self._np
        t = np.linspace(0, 2 * np.pi, n)
        xl, yl = a * np.cos(t), b * np.sin(t)
        ct, st = np.cos(theta), np.sin(theta)
        return cx + xl * ct - yl * st, cy + xl * st + yl * ct

    def _dspin(self, layout, label, lo, hi, val, step, dec):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(100)
        s = QtWidgets.QDoubleSpinBox(); s.setDecimals(int(dec)); s.setRange(float(lo), float(hi))
        s.setSingleStep(float(step)); s.setValue(float(val))
        row.addWidget(lbl); row.addWidget(s, 1); layout.addLayout(row)
        return s

    def _apply_bvm_image(self, img, origin):
        a = self._np.asarray(img, dtype=float)
        self._img.setImage(a)
        v = a[a > 0]
        if v.size:
            self._img.setLevels([float(self._np.percentile(v, 1)),
                                 float(self._np.percentile(v, 99))])
        self._origin = (float(origin[0]), float(origin[1]))   # (y0, x0)
        if self._center_yx is None:
            ec = getattr(self._sc.params, "ellipse_center", None) if self._sc else None
            self._center_yx = (float(ec[0]), float(ec[1])) if ec else self._origin
            self._chk_center.blockSignals(True)
            self._chk_center.setChecked(bool(ec))
            self._chk_center.blockSignals(False)
        self._sync_center_spinboxes()
        qmax = int(min(a.shape) // 2)
        for s in (self._s_r0, self._s_r1):
            s.setMaximum(max(qmax, s.value()))
        self._ell.setData([], [])
        self._fit = None
        self._btn_apply.setEnabled(False)
        self._refresh_view()

    def _reload_bvm(self, *_):
        if self._sc is None:
            self._status.setText("No active scan."); return
        if self._host._busy or self._bvm_pending:
            self._status.setText("BVM reload already running…"); return
        sc = self._sc
        sampling = int(self._s_samp.value())
        use_roi = self._use_roi.isChecked()
        self._bvm_pending = True
        self._status.setText("Rendering BVM…")

        def work():
            if getattr(sc.state, "braggpeaks", None) is None:
                E.load_braggpeaks(sc, log=self._host._console.log)
            E.ensure_pre_step_checkpoint(sc, "ellipse", log=self._host._console.log)
            return E.ellipse_bvm(sc, sampling=sampling, use_roi=use_roi,
                                 log=self._host._console.log)

        def on_done(result):
            self._bvm_pending = False
            if isinstance(result, Exception):
                self._status.setText(f"BVM unavailable: {result}")
                return
            img, origin, bvm = result
            self._cached_bvm = bvm
            self._apply_bvm_image(img, origin)
            self._status.setText(f"BVM ready (sampling={sampling}, ROI={use_roi}). "
                                 f"Adjust q_range, then Fit ellipse.")

        self._host._run_async(work, label=f"Ellipse BVM ({sc.name})", on_done=on_done)

    def _active_center(self):
        if self._chk_center.isChecked() and self._center_yx is not None:
            return self._center_yx
        return self._origin

    def _refresh_view(self, *_):
        y0, x0 = self._origin
        use_custom = self._chk_center.isChecked() and self._center_yx is not None
        cy, cx = self._active_center()
        r0, r1 = self._s_r0.value(), self._s_r1.value()
        self._c0.setData(*self._circle(cx, cy, r0))
        self._c1.setData(*self._circle(cx, cy, r1))
        self._ctr.setData([x0], [y0], size=10, pen=self._pg.mkPen("white"),
                          brush=self._pg.mkBrush("red"))
        if use_custom:
            self._ctr_custom.setData([cx], [cy], size=14, symbol="+",
                                     pen=self._pg.mkPen("#00FF00", width=2.5), brush=None)
        else:
            self._ctr_custom.setData([], [])

    def _sync_center_spinboxes(self):
        cy, cx = self._center_yx if self._center_yx is not None else self._origin  # (y, x)
        for sp, val in ((self._sp_cx, cx), (self._sp_cy, cy)):
            sp.blockSignals(True); sp.setValue(float(val)); sp.blockSignals(False)

    def _on_center_toggled(self, _checked):
        self._refresh_view()

    def _on_center_spin(self, *_):
        self._center_yx = (float(self._sp_cy.value()), float(self._sp_cx.value()))
        self._refresh_view()

    def _reset_center_to_origin(self):
        self._center_yx = self._origin
        self._sync_center_spinboxes()
        self._chk_center.setChecked(False)
        self._refresh_view()

    def _on_click(self, ev):
        p = self._vb.mapSceneToView(ev.scenePos())
        x, y = float(p.x()), float(p.y())
        mods = ev.modifiers()
        if (mods & QtCore.Qt.KeyboardModifier.ShiftModifier
                and not (mods & QtCore.Qt.KeyboardModifier.ControlModifier)
                and ev.button() == QtCore.Qt.MouseButton.LeftButton):
            self._center_yx = (y, x)   # (row, col) — no swap
            self._chk_center.setChecked(True)
            self._sync_center_spinboxes()
            self._refresh_view()
            self._status.setText(
                f"Custom ellipse center set to (row={y:.2f}, col={x:.2f}).")
            return
        if not (mods & QtCore.Qt.KeyboardModifier.ControlModifier):
            return
        y0, x0 = self._active_center()
        r = float(self._np.hypot(x - x0, y - y0))
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._s_r0.setValue(int(round(r)))
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self._s_r1.setValue(int(round(r)))

    def _fit_ellipse(self):
        if self._sc is None or self._host._busy:
            return
        sc = self._sc
        r0, r1 = sorted((self._s_r0.value(), self._s_r1.value()))
        sampling = int(self._s_samp.value())
        use_roi = self._use_roi.isChecked()
        center = self._center_yx if self._chk_center.isChecked() else None
        self._status.setText("Fitting ellipse…")

        def work():
            if getattr(sc.state, "braggpeaks", None) is None:
                E.load_braggpeaks(sc, log=self._host._console.log)
            E.ensure_pre_step_checkpoint(sc, "ellipse", log=self._host._console.log)
            return E.fit_ellipse_preview(
                sc, r0, r1, sampling=sampling, use_roi=use_roi,
                bvm=None, center=center, log=self._host._console.log)

        def on_done(res):
            if isinstance(res, Exception):
                self._status.setText(f"Fit error: {res}"); return
            if not res.get("ok"):
                self._status.setText("Fit failed / no ellipse for this ring.")
                self._fit = None; self._btn_apply.setEnabled(False); return
            self._fit = res
            y0, x0, a, b, theta = res["p_ellipse"]
            self._ell.setData(*self._ellipse_pts(x0, y0, a, b, theta))
            warn = "" if res["ok_scale"] else "   ⚠ scale implausible"
            self._status.setText(
                f"a={a:.2f}  b={b:.2f}  θ={self._np.degrees(theta):.1f}°  "
                f"a/b={res['ab']:.4f}  shift={res['shift']:.2f}px{warn}"
                f"  [pre-ellipse baseline]")
            self._btn_apply.setEnabled(True)
            if res.get("img") is not None and res.get("origin") is not None:
                self._cached_bvm = None
                self._apply_bvm_image(res["img"], res["origin"])

        self._host._run_async(work, label=f"Fit ellipse ({sc.name})", on_done=on_done)

    def _to_params(self):
        """Push the chosen ring / sampling / ROI / center into scan.params (→ parameter table)."""
        if self._sc is None:
            return
        p = self._sc.params
        r0, r1 = sorted((int(self._s_r0.value()), int(self._s_r1.value())))
        p.ellipse_q_range = [r0, r1]
        p.ellipse_sampling = int(self._s_samp.value())
        p.ellipse_use_roi = bool(self._use_roi.isChecked())
        p.ellipse_enabled = True
        if self._chk_center.isChecked() and self._center_yx is not None:
            p.ellipse_center = [round(self._center_yx[0], 3), round(self._center_yx[1], 3)]  # (y, x)
        else:
            p.ellipse_center = None
        self._host._params.reload()
        center_txt = (f" center=({p.ellipse_center[0]:.2f},{p.ellipse_center[1]:.2f})"
                       if p.ellipse_center else " center=bvm.origin")
        self._status.setText(f"Sent to table: q_range=({r0},{r1}) sampling={p.ellipse_sampling} "
                             f"useROI={p.ellipse_use_roi} ellipse_enabled=True{center_txt}")

    def _apply(self):
        if self._sc is None or not self._fit or self._host._busy:
            return
        sc, res = self._sc, self._fit
        self._to_params()                    # also persist the chosen vars to the table

        def work():
            if getattr(sc.state, "braggpeaks", None) is None:
                E.load_braggpeaks(sc, log=self._host._console.log)
            E.ensure_pre_step_checkpoint(sc, "ellipse", log=self._host._console.log)
            E.apply_ellipse_fit(sc, res["p_ellipse"], r_range=(res["r0"], res["r1"]),
                                log=self._host._console.log)

        def on_apply_done(_r):
            self._host._params.reload()
            self._host._update_active_views()
            self._cached_bvm = None
            self._fit = None
            self._btn_apply.setEnabled(False)
            self._reload_bvm()
            self._status.setText("Committed (baseline restored for next fit)")

        self._host._run_async(
            work, label=f"Apply ellipse ({sc.name})", on_done=on_apply_done)

    def _reset(self):
        if self._sc is None or self._host._busy:
            return
        sc = self._sc

        def work():
            if getattr(sc.state, "braggpeaks", None) is None:
                E.load_braggpeaks(sc, log=self._host._console.log)
            E.reset_to_pre_step(sc, "ellipse", log=self._host._console.log)

        self._ell.setData([], []); self._fit = None; self._btn_apply.setEnabled(False)
        self._host._run_async(
            work, label=f"Reset ellipse ({sc.name})",
            on_done=lambda _r: (self._host._params.reload(),
                                self._host._update_active_views(),
                                self._status.setText("Reset to pre-ellipse.")))

    def _on_file_picked(self, sc):
        self._sc = sc
        self._fit = None; self._cached_bvm = None
        self._ell.setData([], []); self._btn_apply.setEnabled(False)
        self._center_yx = None
        self._ctr_custom.setData([], [])
        if sc is not None:
            r0d, r1d = sorted(int(x) for x in (sc.params.ellipse_q_range or [30, 44]))
            self._s_r0.setValue(r0d); self._s_r1.setValue(r1d)
            self._use_roi.setChecked(bool(sc.params.ellipse_use_roi))
            self._chk_center.setChecked(False)   # re-set by _apply_bvm_image from ellipse_center
        self.setWindowTitle(f"Ellipse calibration — {sc.name if sc else ''}")
        self._reload_bvm()

    def _on_prev_calibs_done(self):
        self._reload_bvm()
        self._host._params.reload(); self._host._update_active_views()
        self._status.setText("Upstream calibrations applied — fit the ellipse for this file.")


# ─────────────────────────────────────────────────────────────────────────────
# Q-pixel calibration tool — Update overlay / Test sensitivity / Finalize (REFIT)
# (delegates to the PROVEN fast-mode pipeline functions via engine wrappers)
# ─────────────────────────────────────────────────────────────────────────────

class QPixelDialog(QtWidgets.QDialog):
    """Interactive Q-pixel calibration mirroring the notebook / fast-mode tool:

    · move k_max / bragg_k_power / px (guess) and **Update** the scattering overlay;
    · **Test** = 2N+1 refit sensitivity sweep (guess-vs-fit + residual);
    · **Finalize / REFIT** = crystal.calibrate_pixel_size (the proven fit, with the
      0<px<1 sanity-check) — resets to the pre-Q-pixel checkpoint first so it never
      compounds. The fitted px lands in params.q_px_fitted (the guess q_px is kept).
    """

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from qt_widgets import safe_nav_toolbar
        self._FC, self._safe_tb = FigureCanvasQTAgg, safe_nav_toolbar
        self._host = host
        self._sc = host.active_scan()
        self._canvas = self._tb = None
        self._preview_fig = None
        self._preview_owned = True
        self.setWindowTitle(f"Q-pixel calibration — {self._sc.name if self._sc else ''}")
        self.resize(940, 700)

        lay = QtWidgets.QHBoxLayout(self)
        self._fig_host = QtWidgets.QVBoxLayout()
        fw = QtWidgets.QWidget(); fw.setLayout(self._fig_host)
        lay.addWidget(fw, 1)

        panel = QtWidgets.QWidget(); panel.setMaximumWidth(300)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel("<b>Q-pixel</b> — tune, Test, then Finalize"))
        _add_calib_file_label(self, host, v)
        p = self._sc.params if self._sc else E.CalibrationParams()
        self._sp_px = self._dspin(v, "Q px (guess) Å⁻¹/px", 0.002, 0.2, float(p.q_px), 1e-5, 7)
        self._sp_kmax = self._dspin(v, "k_max (Å⁻¹)", 0.2, 3.0, float(p.q_kmax), 0.01, 2)
        self._sp_kpow = self._dspin(v, "bragg_k_power", 0.1, 6.0, float(p.q_kpow), 0.1, 2)
        self._use_roi = QtWidgets.QCheckBox("Use ROI"); self._use_roi.setChecked(bool(p.q_use_roi))
        v.addWidget(self._use_roi)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Test span:"))
        self._n_fig = QtWidgets.QComboBox(); self._n_fig.addItems(["3", "5", "7"])
        self._n_fig.setCurrentText("7")
        row.addWidget(self._n_fig)
        v.addLayout(row)
        self._sp_step = self._dspin(v, "test step (Δpx)", 1e-6, 1e-3, 1e-4, 1e-5, 7)

        for txt, fn, tip in (("Update", self._update, "Refresh the scattering overlay (preview only)."),
                             ("Test", self._test, "2N+1 refit sensitivity sweep around the px guess."),
                             (LBL_TO_TABLE, self._to_params, TIP_TO_TABLE_QPIXEL),
                             (LBL_APPLY_CALIB, self._finalize, "Crystal pixel-size refit; resets pre-Q-pixel baseline first."),
                             (LBL_RESET, self._reset, TIP_RESET)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn)
            if tip:
                b.setToolTip(tip)
            v.addWidget(b)
        v.addStretch(1)
        self._status = QtWidgets.QLabel("Ready"); self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        v.addWidget(bb)
        lay.addWidget(panel)
        if self._sc is not None:
            self._update()

    def _dspin(self, layout, label, lo, hi, val, step, dec):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(150)
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(int(dec)); s.setRange(float(lo), float(hi))
        s.setSingleStep(float(step)); s.setValue(float(val))
        row.addWidget(lbl); row.addWidget(s, 1)
        layout.addLayout(row)
        return s

    def _show_fig(self, fig, *, owned: bool = True):
        if fig is None:
            self._status.setText("No figure produced."); return
        prev = getattr(self, "_preview_fig", None)
        if prev is not None and prev is not fig and getattr(self, "_preview_owned", True):
            E.close_figure(prev)
        if self._canvas is not None:
            self._fig_host.removeWidget(self._canvas); self._canvas.setParent(None)
            self._canvas.deleteLater(); self._canvas = None
        if self._tb is not None:
            self._fig_host.removeWidget(self._tb); self._tb.setParent(None)
            self._tb.deleteLater(); self._tb = None
        self._preview_fig = fig
        self._preview_owned = bool(owned)
        self._canvas = self._FC(fig)
        self._tb = self._safe_tb(self._canvas, self)
        self._fig_host.addWidget(self._tb); self._fig_host.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def closeEvent(self, ev) -> None:
        if getattr(self, "_preview_owned", True) and getattr(self, "_preview_fig", None):
            E.close_figure(self._preview_fig)
        self._preview_fig = None
        self._host._maybe_tidy_figures()
        super().closeEvent(ev)

    def _vals(self):
        return dict(px=self._sp_px.value(), k_max=self._sp_kmax.value(),
                    kpow=self._sp_kpow.value(), use_roi=self._use_roi.isChecked())

    def _update(self):
        if self._sc is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            vv = self._vals()
            fig = E.q_pixel_overlay(self._sc, px=vv["px"], k_max=vv["k_max"],
                                    kpow=vv["kpow"], use_roi=vv["use_roi"],
                                    log=self._host._console.log)
            self._show_fig(fig)
            self._status.setText(f"Overlay: px={vv['px']:.7g}  k_max={vv['k_max']:.2f}  "
                                 f"kpow={vv['kpow']:.2f}  ROI={vv['use_roi']}")
        except Exception as exc:
            self._status.setText(f"Update error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._host._maybe_tidy_figures()

    def _test(self):
        if self._sc is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            vv = self._vals()
            res = E.q_pixel_test(self._sc, px0=vv["px"], test_step=self._sp_step.value(),
                                 n_figures=int(self._n_fig.currentText()), k_max=vv["k_max"],
                                 kpow=vv["kpow"], use_roi=vv["use_roi"],
                                 log=self._host._console.log)
            self._show_fig(res.get("summary_figure") or (res.get("figures") or [None])[-1])
            nfit = len(res.get("px_fit", []))
            self._status.setText(f"Test: {nfit} fit(s), {len(res.get('failures', []))} fail(s) "
                                 f"(does NOT change the applied calibration).")
        except Exception as exc:
            self._status.setText(f"Test error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._host._maybe_tidy_figures()

    def _to_params(self):
        """Push the chosen guess px / k_max / bragg_k_power / ROI into scan.params
        (→ parameter table) so a later Compute uses exactly what was tuned here."""
        if self._sc is None:
            return
        vv = self._vals()
        p = self._sc.params
        p.q_px = float(vv["px"])
        p.q_kmax = float(vv["k_max"])
        p.q_kpow = float(vv["kpow"])
        p.q_use_roi = bool(vv["use_roi"])
        self._host._params.reload()
        self._status.setText(f"Sent to table: q_px(guess)={p.q_px:.7g} k_max={p.q_kmax:.2f} "
                             f"bragg_k_power={p.q_kpow:.2f} useROI={p.q_use_roi}")

    def _finalize(self):
        if self._sc is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            vv = self._vals()
            self._to_params()           # persist the tuned vars to the table too
            E.ensure_pre_step_checkpoint(self._sc, "qpixel", log=self._host._console.log)
            res = E.q_pixel_finalize(self._sc, px_guess=vv["px"], k_max=vv["k_max"],
                                     kpow=vv["kpow"], use_roi=vv["use_roi"],
                                     log=self._host._console.log)
            self._show_fig(res.get("figure"), owned=False)
            self._status.setText(f"REFIT done: guess={vv['px']:.7g} → fitted="
                                 f"{res.get('px_fit', float('nan')):.7g} (applied).")
            self._host._params.reload(); self._host._update_active_views()
        except Exception as exc:
            self._status.setText(f"Finalize error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _show_registered_qpixel_fig(self) -> bool:
        """Show the stored FIT figure, if present (not the guess overlay)."""
        if self._sc is None:
            return False
        fig = E.collect_figures(self._sc).get("q_pixel")
        if fig is None:
            return False
        self._show_fig(fig, owned=False)
        px = float(getattr(self._sc.params, "q_px_fitted", None) or self._sp_px.value())
        self._status.setText(f"Showing FIT figure (px={px:.7g}). "
                             f"'Update overlay' shows the guess scattering preview.")
        return True

    def _reset(self):
        if self._sc is None:
            return
        if getattr(self._sc.state, "braggpeaks", None) is None:
            E.load_braggpeaks(self._sc, log=self._host._console.log)
        E.reset_to_pre_step(self._sc, "qpixel", log=self._host._console.log)
        self._host._params.reload(); self._host._update_active_views()
        self._status.setText("Reset to pre-Q-pixel.")

    def _on_file_picked(self, sc):
        self._sc = sc
        if sc is not None:
            p = sc.params
            self._sp_px.setValue(float(p.q_px)); self._sp_kmax.setValue(float(p.q_kmax))
            self._sp_kpow.setValue(float(p.q_kpow)); self._use_roi.setChecked(bool(p.q_use_roi))
        self.setWindowTitle(f"Q-pixel calibration — {sc.name if sc else ''}")
        self._status.setText("File selected. Use Prep upstream, then Update / Test / Commit.")

    def _on_prev_calibs_done(self):
        if not self._show_registered_qpixel_fig():
            self._update()
        self._host._params.reload(); self._host._update_active_views()


# ─────────────────────────────────────────────────────────────────────────────
# Basis tuner — move the basis vars and see the choose_basis_vectors preview
# (like the Detect tuner; runs on the GUI thread because the preview uses pyplot)
# ─────────────────────────────────────────────────────────────────────────────

class BasisDialog(QtWidgets.QDialog):
    """Interactive basis calibration: move the basis-finding vars (minSpacing,
    minAbsoluteIntensity threshold, maxNumPeaks, edgeBoundary, vis vmin/vmax, QR
    rotation/flip, optional manual g1/g2 indices) and see the choose_basis_vectors
    preview update. 'Finalize' keeps the chosen basis (resets to pre-basis first)."""

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from qt_widgets import safe_nav_toolbar
        self._FC, self._safe_tb = FigureCanvasQTAgg, safe_nav_toolbar
        self._host = host
        self._sc = host.active_scan()
        self._canvas = self._tb = None
        self._preview_fig = None
        self._preview_owned = True
        self._snapped = False
        self.setWindowTitle(f"Basis calibration — {self._sc.name if self._sc else ''}")
        self.resize(980, 720)

        lay = QtWidgets.QHBoxLayout(self)
        self._fig_host = QtWidgets.QVBoxLayout()
        fw = QtWidgets.QWidget(); fw.setLayout(self._fig_host)
        lay.addWidget(fw, 1)

        panel = QtWidgets.QWidget(); panel.setMaximumWidth(310)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel("<b>Basis</b> — move vars, see choose_basis_vectors"))
        _add_calib_file_label(self, host, v)
        p = self._sc.params if self._sc else E.CalibrationParams()
        self._ispin = {}
        self._ispin["min_spacing"] = self._ispin_row(v, "minSpacing", 0, 100, int(p.min_spacing))
        self._ispin["min_absolute_intensity"] = self._ispin_row(
            v, "minAbsoluteIntensity", 0, 1000, int(p.min_absolute_intensity), step=5)
        self._ispin["max_num_peaks"] = self._ispin_row(v, "maxNumPeaks", 1, 300, int(p.max_num_peaks))
        self._ispin["edge_boundary"] = self._ispin_row(v, "edgeBoundary", 0, 100, int(p.edge_boundary))
        self._sp_vmin = self._dspin_row(v, "vis vmin", 0.0, 1.0, float(p.vis_vmin), 0.005, 3)
        self._sp_vmax = self._dspin_row(v, "vis vmax", 0.0, 1.0, float(p.vis_vmax), 0.005, 3)
        self._sp_qr = self._dspin_row(v, "QR rotation (deg)", -360.0, 360.0,
                                      float(p.qr_rotation), 0.5, 1)
        self._cb_flip = QtWidgets.QCheckBox("QR flip"); self._cb_flip.setChecked(bool(p.qr_flip))
        v.addWidget(self._cb_flip)
        self._cb_manual = QtWidgets.QCheckBox("Manual g1/g2 indices")
        self._cb_manual.setChecked(bool(p.basis_manual_enabled))
        v.addWidget(self._cb_manual)
        self._ispin["index_origin"] = self._ispin_row(v, "index_origin", 0, 50, int(p.index_origin))
        self._ispin["index_g1"] = self._ispin_row(v, "index_g1", 0, 50, int(p.index_g1))
        self._ispin["index_g2"] = self._ispin_row(v, "index_g2", 0, 50, int(p.index_g2))
        self._auto = QtWidgets.QCheckBox("Auto-update on change"); self._auto.setChecked(True)
        v.addWidget(self._auto)

        # Update = recompute the preview (no commit); Apply = commit the basis to the
        # state + push the vars to the parameter table (like the notebook's Update/Apply).
        for txt, fn, tip in (("Update", self._update, "Recompute basis preview (no commit)."),
                             (LBL_TO_TABLE, self._to_params, TIP_TO_TABLE_BASIS),
                             (LBL_APPLY_CALIB, self._apply_basis, "Write basis calibration to braggpeaks."),
                             (LBL_RESET, self._reset, TIP_RESET)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn)
            if tip:
                b.setToolTip(tip)
            v.addWidget(b)
        v.addStretch(1)
        self._status = QtWidgets.QLabel("Ready"); self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        v.addWidget(bb)
        lay.addWidget(panel)

        # debounced auto-update: the heavy braggpeaks load runs in a worker; the
        # pyplot preview is built back on the GUI thread (see _update / on_done).
        self._timer = QtCore.QTimer(self); self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._update)
        for w in list(self._ispin.values()) + [self._sp_vmin, self._sp_vmax, self._sp_qr]:
            w.valueChanged.connect(self._schedule)
        for c in (self._cb_flip, self._cb_manual):
            c.toggled.connect(self._schedule)
        if self._sc is not None:
            self._update()

    def _ispin_row(self, layout, label, lo, hi, val, step=1):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(150)
        s = QtWidgets.QSpinBox(); s.setRange(int(lo), int(hi)); s.setSingleStep(int(step))
        s.setValue(int(val))
        row.addWidget(lbl); row.addWidget(s, 1); layout.addLayout(row)
        return s

    def _dspin_row(self, layout, label, lo, hi, val, step, dec):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(150)
        s = QtWidgets.QDoubleSpinBox(); s.setDecimals(int(dec)); s.setRange(float(lo), float(hi))
        s.setSingleStep(float(step)); s.setValue(float(val))
        row.addWidget(lbl); row.addWidget(s, 1); layout.addLayout(row)
        return s

    def _schedule(self, *_):
        if self._auto.isChecked():
            self._timer.start(350)

    def _collect(self):
        p = self._sc.params
        p.min_spacing = int(self._ispin["min_spacing"].value())
        p.min_absolute_intensity = int(self._ispin["min_absolute_intensity"].value())
        p.max_num_peaks = int(self._ispin["max_num_peaks"].value())
        p.edge_boundary = int(self._ispin["edge_boundary"].value())
        vmin, vmax = float(self._sp_vmin.value()), float(self._sp_vmax.value())
        if vmin > vmax:                      # enforce vmin <= vmax
            vmin, vmax = vmax, vmin
            self._sp_vmin.setValue(vmin); self._sp_vmax.setValue(vmax)
        p.vis_vmin, p.vis_vmax = vmin, vmax
        p.qr_rotation = float(self._sp_qr.value())
        p.qr_flip = bool(self._cb_flip.isChecked())
        p.basis_manual_enabled = bool(self._cb_manual.isChecked())
        p.index_origin = int(self._ispin["index_origin"].value())
        p.index_g1 = int(self._ispin["index_g1"].value())
        p.index_g2 = int(self._ispin["index_g2"].value())

    def _show_fig(self, fig, *, owned: bool = True):
        if fig is None:
            self._status.setText("No preview figure produced."); return
        prev = getattr(self, "_preview_fig", None)
        if prev is not None and prev is not fig and getattr(self, "_preview_owned", True):
            E.close_figure(prev)
        if self._canvas is not None:
            self._fig_host.removeWidget(self._canvas); self._canvas.setParent(None)
            self._canvas.deleteLater(); self._canvas = None
        if self._tb is not None:
            self._fig_host.removeWidget(self._tb); self._tb.setParent(None)
            self._tb.deleteLater(); self._tb = None
        self._preview_fig = fig
        self._preview_owned = bool(owned)
        self._canvas = self._FC(fig)
        self._tb = self._safe_tb(self._canvas, self)
        self._fig_host.addWidget(self._tb); self._fig_host.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def closeEvent(self, ev) -> None:
        if getattr(self, "_preview_owned", True) and getattr(self, "_preview_fig", None):
            E.close_figure(self._preview_fig)
        self._preview_fig = None
        self._host._maybe_tidy_figures()
        super().closeEvent(ev)

    def _to_params(self):
        """Push the basis vars into scan.params (→ parameter table) without re-running
        the preview (the controls are already collected into params by _collect)."""
        if self._sc is None:
            return
        self._collect()
        self._host._params.reload()
        self._status.setText("Sent to table: basis params updated.")

    def _update(self, *, then=None):
        # The heavy braggpeaks .h5 load + pre-basis snapshot run in a worker (no pyplot,
        # thread-safe); the basis_preview (touches pyplot) + canvas build run back on the
        # GUI thread in on_done. ``then`` lets synchronous callers chain work AFTER the
        # preview is shown (the recompute used to be inline).
        if self._sc is None:
            if then is not None:
                then()
            return
        if self._host._busy:                # a heavy op is already running — skip this tick
            return
        sc = self._sc

        def work():
            if not self._snapped:           # one-time pre-basis baseline (for Reset)
                if getattr(sc.state, "braggpeaks", None) is None:
                    E.load_braggpeaks(sc, log=self._host._console.log)
                sc.cal_checkpoints.setdefault("pre_basis", E.snapshot_calibration(sc))
                self._snapped = True

        def on_done(_r):
            try:                            # pyplot preview + canvas: GUI thread only
                self._collect()
                fig = E.basis_preview(sc, log=self._host._console.log)
                self._show_fig(fig)
                self._status.setText(
                    f"Preview: minAbsInt={sc.params.min_absolute_intensity} "
                    f"maxPeaks={sc.params.max_num_peaks} QRrot={sc.params.qr_rotation:.1f}° "
                    f"flip={sc.params.qr_flip} manual={sc.params.basis_manual_enabled}")
            except Exception as exc:
                self._status.setText(f"Preview error: {exc}")
            if then is not None:
                then()

        self._host._run_async(work, label=f"Basis preview ({sc.name})", on_done=on_done)

    def _apply_basis(self):
        if self._sc is None:
            return
        self._update(then=self._commit_basis)   # recompute (async) → commit in on_done

    def _commit_basis(self):
        """Runs on the GUI thread after _update's preview is shown: push the vars to the
        table and register the committed basis figure (was inline in _apply_basis)."""
        if self._sc is None:
            return
        self._to_params()                   # push the vars to the parameter table
        fig = getattr(self, "_preview_fig", None)
        if fig is not None:
            E.register_figure(self._sc, "basis", fig, force=True)
            self._preview_owned = False
        self._host._params.reload(); self._host._update_active_views()
        self._status.setText("Basis committed (chosen basis + vars saved).")

    def _reset(self):
        if self._sc is None or self._host._busy:
            return
        sc = self._sc

        def work():
            if getattr(sc.state, "braggpeaks", None) is None:
                E.load_braggpeaks(sc, log=self._host._console.log)
            E.reset_to_pre_step(sc, "basis", log=self._host._console.log)

        self._host._run_async(
            work, label=f"Reset basis ({sc.name})",
            on_done=lambda _r: (self._host._params.reload(),
                                self._host._update_active_views(),
                                self._status.setText("Reset to pre-basis.")))

    def _on_file_picked(self, sc):
        self._sc = sc
        self._snapped = False
        self._preview_fig = None
        self._preview_owned = True
        if sc is not None:
            p = sc.params
            self._ispin["min_spacing"].setValue(int(p.min_spacing))
            self._ispin["min_absolute_intensity"].setValue(int(p.min_absolute_intensity))
            self._ispin["max_num_peaks"].setValue(int(p.max_num_peaks))
            self._ispin["edge_boundary"].setValue(int(p.edge_boundary))
            self._sp_qr.setValue(float(p.qr_rotation)); self._cb_flip.setChecked(bool(p.qr_flip))
        self.setWindowTitle(f"Basis calibration — {sc.name if sc else ''}")
        self._status.setText("File selected. Use Prep upstream, then Update / Commit.")

    def _on_prev_calibs_done(self):
        self._update(then=lambda: (self._host._params.reload(),
                                   self._host._update_active_views()))


# ─────────────────────────────────────────────────────────────────────────────
# Custom calibration crystal — pick element(s) + structure + lattice param; the
# atomic-positions array is GENERATED (no hand-written coordinates)
# ─────────────────────────────────────────────────────────────────────────────

class CrystalEditorDialog(QtWidgets.QDialog):
    """Define the Q-pixel calibration crystal: element(s) + structure type + lattice
    parameter. The positions array is generated from the structure (the user never
    writes coordinates). Single-species structures use element A everywhere; two-
    species (zincblende/rocksalt/CsCl) put element A / B on the two sublattices."""

    _LABELS = [("Diamond cubic (Si, Ge…)", "diamond"), ("FCC (Au, Al, Cu…)", "fcc"),
               ("BCC (Fe, W…)", "bcc"), ("Simple cubic", "simple_cubic"),
               ("Zincblende (2 elem: GaAs…)", "zincblende"),
               ("Rocksalt (2 elem: NaCl…)", "rocksalt"),
               ("CsCl (2 elem)", "cscl")]

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        self._host = host
        self._sc = host.active_scan()
        self.setWindowTitle(f"Calibration crystal — {self._sc.name if self._sc else ''}")
        self.resize(560, 520)
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            "<b>Custom calibration crystal</b> — element(s) + structure + lattice "
            "constant.\nThe atomic-positions array is generated for you."))
        form = QtWidgets.QFormLayout()
        self._struct = QtWidgets.QComboBox()
        for lbl, dat in self._LABELS:
            self._struct.addItem(lbl, dat)
        self._struct.currentIndexChanged.connect(self._on_struct)
        form.addRow("Structure", self._struct)
        self._el_a = QtWidgets.QLineEdit("Si")
        self._el_a.setToolTip("Element symbol (Si, Ge, Au…) or atomic number Z")
        form.addRow("Element A", self._el_a)
        self._el_b = QtWidgets.QLineEdit("Ge")
        self._el_b.setToolTip("Second element (only for zincblende/rocksalt/CsCl)")
        form.addRow("Element B", self._el_b)
        self._a = QtWidgets.QDoubleSpinBox()
        self._a.setDecimals(4); self._a.setRange(0.5, 50.0); self._a.setSingleStep(0.001)
        self._a.setValue(5.431)
        form.addRow("Lattice a (Å)", self._a)
        v.addLayout(form)
        # SiGe helper: Vegard interpolation of the lattice constant
        veg = QtWidgets.QHBoxLayout()
        self._vegard = QtWidgets.QPushButton("Vegard a(Si,Ge): x=")
        self._vegard.setToolTip("Set a = (1-x)·a_Si + x·a_Ge  (a_Si=5.431, a_Ge=5.658)")
        self._x = QtWidgets.QDoubleSpinBox(); self._x.setDecimals(3); self._x.setRange(0.0, 1.0)
        self._x.setSingleStep(0.05); self._x.setValue(0.30)
        self._vegard.clicked.connect(self._apply_vegard)
        veg.addWidget(self._vegard); veg.addWidget(self._x); veg.addStretch(1)
        v.addLayout(veg)

        brow = QtWidgets.QHBoxLayout()
        b_gen = QtWidgets.QPushButton("Generate / preview"); b_gen.clicked.connect(self._generate)
        b_use = QtWidgets.QPushButton("Apply (file)"); b_use.clicked.connect(lambda: self._use(False))
        b_all = QtWidgets.QPushButton("Apply (all)"); b_all.clicked.connect(lambda: self._use(True))
        b_cif = QtWidgets.QPushButton("Load CIF…"); b_cif.clicked.connect(self._load_cif)
        b_cif.setToolTip("Use a CIF file as the calibration crystal for Q-pixel.")
        for b in (b_gen, b_use, b_all, b_cif):
            brow.addWidget(b)
        v.addLayout(brow)
        self._preview = QtWidgets.QPlainTextEdit(); self._preview.setReadOnly(True)
        self._preview.setStyleSheet("font-family:monospace; font-size:10px;")
        v.addWidget(self._preview, 1)
        self._status = QtWidgets.QLabel(""); self._status.setStyleSheet("color:#1565C0;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept); v.addWidget(bb)

        self._load_existing()
        self._on_struct()
        self._generate()

    def _load_existing(self):
        cc = (self._sc.params.custom_crystal if self._sc else None) or {}
        if cc:
            st = cc.get("structure")
            if st:
                i = self._struct.findData(st)
                if i >= 0:
                    self._struct.setCurrentIndex(i)
            self._a.setValue(float(cc.get("a_lat", 5.431)))
            an = cc.get("atom_num")
            try:
                import engine as _E
                inv = {z: s for s, z in _E.ELEMENT_Z.items()}
                if isinstance(an, (list, tuple)) and an:
                    self._el_a.setText(inv.get(int(an[0]), str(an[0])))
                    self._el_b.setText(inv.get(int(an[-1]), str(an[-1])))
                elif an is not None:
                    self._el_a.setText(inv.get(int(an), str(an)))
            except Exception:
                pass

    def _on_struct(self, *_):
        two = self._struct.currentData() in E.TWO_SPECIES_STRUCTURES
        self._el_b.setEnabled(two)

    def _apply_vegard(self):
        x = float(self._x.value())
        self._a.setValue((1.0 - x) * 5.431 + x * 5.658)
        self._status.setText(f"a set by Vegard (x={x:.3f}) → {self._a.value():.4f} Å")

    def _build(self):
        return E.build_custom_crystal(self._struct.currentData(), self._a.value(),
                                      self._el_a.text(),
                                      self._el_b.text() if self._el_b.isEnabled() else None)

    def _generate(self):
        try:
            cc = self._build()
        except Exception as exc:
            self._status.setText(f"Error: {exc}"); return None
        pos = cc["positions"]; an = cc["atom_num"]
        lines = [f"structure = {cc['structure']}", f"a_lat = {cc['a_lat']:.4f} Å",
                 f"atom_num = {an}", f"{len(pos)} sites (fractional positions):"]
        for p in pos:
            lines.append(f"  [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
        self._preview.setPlainText("\n".join(lines))
        self._status.setText("Generated. 'Use for this file' / 'Use for ALL files' to apply.")
        return cc

    def _use(self, all_files):
        cc = self._generate()
        if cc is None or self._sc is None:
            return
        targets = self._host._scans if all_files else [self._sc]
        for s in targets:
            s.params.cal_crystal = "Custom"
            s.params.custom_crystal = dict(cc)
        self._host._params.reload()
        self._status.setText(f"Applied to {len(targets)} file(s) (cal_crystal=Custom).")

    def _load_cif(self):
        if self._sc is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load crystal CIF", "", "CIF files (*.cif);;All files (*.*)")
        if not path:
            return
        try:
            info = E.load_crystal_from_cif(path)
        except Exception as exc:
            self._status.setText(f"CIF load failed: {exc}")
            return
        self._sc.params.cal_crystal = "CIF"
        self._sc.params.cif_path = info.path
        self._host._params.reload()
        lines = [
            f"CIF = {info.path}",
            f"name = {info.cal.name}",
            f"a_lat = {info.cal.a_lat:.4f} Å",
            f"sites = {len(info.cal.positions)}",
            f"cubic = {info.is_cubic}",
        ]
        if info.warning:
            lines.append(f"WARNING: {info.warning}")
        self._preview.setPlainText("\n".join(lines))
        self._status.setText(
            f"Applied CIF to this file (cal_crystal=CIF, a={info.cal.a_lat:.4f} Å)."
            + (" Non-cubic — Index/Q-pixel use effective a." if info.warning else "")
        )


# ─────────────────────────────────────────────────────────────────────────────
# Virtualization — build the virtual-images .h5 (ADF/BF/DP mean/max) from the raw
# datacube (port of the single-mode virtualization_window.py)
# ─────────────────────────────────────────────────────────────────────────────

def _show_vimg_preview(parent, imgs: dict) -> None:
    """Show computed virtual images (ADF / BF) in a non-modal dialog immediately
    after calculation — so the user doesn't need to save the .h5 to see the result.

    imgs: {label: 2D numpy array}  e.g. {"ADF": arr, "BF": arr}
    """
    import numpy as _np
    import pyqtgraph as pg

    dlg = QtWidgets.QDialog(parent)
    _enable_minmax(dlg)
    dlg.setWindowTitle("Virtual image preview")
    dlg.resize(460 * max(len(imgs), 1), 500)
    pg.setConfigOptions(imageAxisOrder="row-major")

    outer = QtWidgets.QVBoxLayout(dlg)
    imgs_row = QtWidgets.QHBoxLayout()

    for label, arr in imgs.items():
        col = QtWidgets.QVBoxLayout()
        col.addWidget(QtWidgets.QLabel(f"<b>{label}</b>"))
        glw = pg.GraphicsLayoutWidget()
        vb = glw.addViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)
        img_item = pg.ImageItem()
        vb.addItem(img_item)
        data = _np.asarray(arr, dtype=float)
        lo = float(_np.nanpercentile(data, 1))
        hi = float(_np.nanpercentile(data, 99))
        img_item.setImage(data, levels=(lo, hi))
        col.addWidget(glw, 1)
        imgs_row.addLayout(col, 1)

    outer.addLayout(imgs_row, 1)
    bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    bb.rejected.connect(dlg.accept)
    outer.addWidget(bb)
    dlg.show()


class VirtualizationDialog(QtWidgets.QDialog):
    """Build the virtual-images .h5 from the raw datacube: DP mean/max + probe →
    place BF (circle) + ADF (annulus) detectors (click center, sliders for radii,
    or 'Init from probe') → compute ADF/BF → save .h5. Heavy passes run in the
    host's worker so the UI stays responsive."""

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        import numpy as np
        import pyqtgraph as pg
        self._np, self._pg = np, pg
        self._host = host
        self._sc = host.active_scan()
        self._dp_mean = None
        self._dp_max = None
        self._alpha = None          # alpha of this scan's DP (get_probe_size)
        self._alpha_vac = None      # alpha of the vacuum probe (the kernel)
        self._vac_probe = None      # vacuum probe image (shown if no DP yet)
        self._qshape = None
        self._det_info = {}
        self._cx = 128.0   # center col (x = qy)
        self._cy = 128.0   # center row (y = qx)
        self.setWindowTitle(f"Create ADF / BF / DP  —  {self._sc.name if self._sc else ''}")
        self.resize(1040, 760)

        lay = QtWidgets.QHBoxLayout(self)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._glw = pg.GraphicsLayoutWidget()
        self._vb = self._glw.addViewBox(); self._vb.setAspectLocked(True); self._vb.invertY(True)
        self._img = pg.ImageItem(); self._vb.addItem(self._img)
        self._c_bf = pg.PlotDataItem(pen=pg.mkPen("#00E5FF", width=2))
        self._c_in = pg.PlotDataItem(pen=pg.mkPen("#FFEB3B", width=2, style=QtCore.Qt.PenStyle.DashLine))
        self._c_out = pg.PlotDataItem(pen=pg.mkPen("#FF7043", width=2))
        self._ctr = pg.ScatterPlotItem()
        for it in (self._c_bf, self._c_in, self._c_out, self._ctr):
            it.setZValue(10); self._vb.addItem(it)
        # INTERACTIVE detectors: drag to move the shared center / drag a ring edge to
        # resize it (mirrors the Tk tool). Left-drag is captured via an event filter on
        # the view's viewport (so pyqtgraph's pan doesn't steal it); wheel still zooms.
        self._drag = None
        self._glw.viewport().installEventFilter(self)
        lay.addWidget(self._glw, 1)

        panel = QtWidgets.QWidget(); panel.setMaximumWidth(320)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel("<b>Create virtual images</b>"))
        # choose which loaded file to build from (only those with a raw 4D path)
        frow = QtWidgets.QHBoxLayout()
        frow.addWidget(QtWidgets.QLabel("File:"))
        self._file_combo = QtWidgets.QComboBox()
        for i, s in enumerate(host._scans):
            tag = "" if s.raw_path else "  (no raw 4D)"
            self._file_combo.addItem(f"{s.name}{tag}", i)
        _def = host._active if (0 <= host._active < len(host._scans)) else 0
        if host._scans:
            self._file_combo.setCurrentIndex(_def)
            self._sc = host._scans[_def]          # keep self._sc in sync with the combo
        self._file_combo.currentIndexChanged.connect(self._on_file_changed)
        frow.addWidget(self._file_combo, 1)
        v.addLayout(frow)
        v.addWidget(QtWidgets.QLabel("1) Compute DP+probe · 2) place detectors\n"
                                     "3) Compute ADF/BF · 4) Save .h5"))
        moderow = QtWidgets.QHBoxLayout()
        moderow.addWidget(QtWidgets.QLabel("Compute:"))
        self._rb_mode_mean = QtWidgets.QRadioButton("Mean only")
        self._rb_mode_max = QtWidgets.QRadioButton("Max only")
        self._rb_mode_both = QtWidgets.QRadioButton("Both (slower)")
        self._rb_mode_both.setChecked(True)
        for rb in (self._rb_mode_mean, self._rb_mode_max, self._rb_mode_both):
            rb.setToolTip("Which full-4D pass(es) to run — skipping one halves the "
                          "heavy compute time for large files.")
            moderow.addWidget(rb)
        moderow.addStretch(1)
        v.addLayout(moderow)
        b_dp = QtWidgets.QPushButton("Compute DP + probe (heavy)")
        b_dp.clicked.connect(self._compute_dp); v.addWidget(b_dp)
        self._b_dp = b_dp
        # display mean/max
        drow = QtWidgets.QHBoxLayout()
        self._rb_mean = QtWidgets.QRadioButton("Mean DP"); self._rb_mean.setChecked(True)
        self._rb_max = QtWidgets.QRadioButton("Max DP")
        self._rb_mean.toggled.connect(lambda _o: self._redraw_dp())
        drow.addWidget(self._rb_mean); drow.addWidget(self._rb_max); drow.addStretch(1)
        v.addLayout(drow)
        self._probe_lbl = QtWidgets.QLabel("Probe: (not computed)")
        self._probe_lbl.setStyleSheet("font-family:monospace; font-size:10px; color:#0D47A1;")
        v.addWidget(self._probe_lbl)
        # mouse-tool selector (Auto: near a ring edge → resize, else → move center)
        mtool = QtWidgets.QHBoxLayout()
        mtool.addWidget(QtWidgets.QLabel("Mouse:"))
        self._mode_combo = QtWidgets.QComboBox()
        for lbl, dat in (("Auto (edge=resize, else move)", "auto"),
                         ("Move center", "move"), ("Resize BF", "bf"),
                         ("Resize ADF inner", "adf_inner"), ("Resize ADF outer", "adf_outer")):
            self._mode_combo.addItem(lbl, dat)
        mtool.addWidget(self._mode_combo, 1)
        v.addLayout(mtool)
        v.addWidget(QtWidgets.QLabel("Drag on the DP: move the center or a ring edge."))
        self._s_bf = self._slider(v, "BF radius", 0, 256, 20)
        self._s_in = self._slider(v, "ADF inner", 0, 256, 30)
        self._s_out = self._slider(v, "ADF outer", 0, 256, 90)
        for s in (self._s_bf, self._s_in, self._s_out):
            s.valueChanged.connect(self._redraw_overlays)
        # detector preset: radii = ratio x alpha (default BF = a, HAADF 2a..8a)
        grp = QtWidgets.QGroupBox("Detectors from alpha (default, editable)")
        g = QtWidgets.QVBoxLayout(grp)
        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(QtWidgets.QLabel("alpha from:"))
        self._alpha_src = QtWidgets.QComboBox()
        self._alpha_src.addItem("Vacuum probe (kernel)", "vacuum")
        self._alpha_src.addItem("This scan's DP", "dp")
        self._alpha_src.setToolTip(
            "Vacuum: alpha = radius of the vacuum probe (no sample scattering) — the\n"
            "same kernel used for disk detection. Computed once per vacuum file and\n"
            "cached (memory + ~/.fast4d/probe_cache), so other scans reuse it.")
        self._alpha_src.currentIndexChanged.connect(lambda _i: self._apply_mult())
        arow.addWidget(self._alpha_src, 1)
        g.addLayout(arow)
        self._b_vac = QtWidgets.QPushButton("Load vacuum probe (cached)")
        self._b_vac.setToolTip("Gets alpha (and a start center) from the vacuum file. "
                               "Uses the cache if this vacuum was already processed — "
                               "no vacuum reload, no 4D pass over the scan.")
        self._b_vac.clicked.connect(self._load_vacuum_probe)
        g.addWidget(self._b_vac)
        mrow = QtWidgets.QHBoxLayout()
        self._m_bf = self._dspin(0.0, 20.0, E.DETECTOR_RATIOS["bf"])
        self._m_in = self._dspin(0.0, 20.0, E.DETECTOR_RATIOS["adf_inner"])
        self._m_out = self._dspin(0.0, 40.0, E.DETECTOR_RATIOS["adf_outer"])
        for sp in (self._m_bf, self._m_in, self._m_out):
            sp.setMinimumWidth(52); sp.setMaximumWidth(62); sp.setDecimals(1)
        mrow.setSpacing(3)
        mrow.addWidget(QtWidgets.QLabel("BF×")); mrow.addWidget(self._m_bf)
        mrow.addWidget(QtWidgets.QLabel("in×")); mrow.addWidget(self._m_in)
        mrow.addWidget(QtWidgets.QLabel("out×")); mrow.addWidget(self._m_out)
        g.addLayout(mrow)
        crow = QtWidgets.QHBoxLayout()
        self._clamp = QtWidgets.QCheckBox("Clamp outer to detector edge")
        self._clamp.setChecked(True)
        self._clamp.setToolTip("Keep the HAADF ring complete (a ring cut by the detector "
                               "edge is not azimuthally symmetric).")
        crow.addWidget(self._clamp)
        g.addLayout(crow)
        mr = QtWidgets.QHBoxLayout()
        mr.addWidget(QtWidgets.QLabel("alpha (mrad, optional):"))
        self._alpha_mrad = QtWidgets.QDoubleSpinBox()
        self._alpha_mrad.setRange(0.0, 200.0); self._alpha_mrad.setDecimals(2)
        self._alpha_mrad.setSpecialValueText("unknown")
        self._alpha_mrad.setToolTip("Convergence semi-angle. Only used to show the radii "
                                    "in mrad (px -> mrad = alpha_mrad / alpha_px).")
        try:
            self._alpha_mrad.setValue(float(QtCore.QSettings("Fast4D", "virtualization")
                                             .value("alpha_mrad", 0.0)))
        except Exception:
            pass
        self._alpha_mrad.valueChanged.connect(self._on_alpha_mrad)
        mr.addWidget(self._alpha_mrad)
        g.addLayout(mr)
        bm = QtWidgets.QHBoxLayout()
        b_mult = QtWidgets.QPushButton("Apply ratios")
        b_mult.clicked.connect(self._apply_mult); bm.addWidget(b_mult)
        b_def = QtWidgets.QPushButton("Reset 1 / 2 / 8")
        b_def.clicked.connect(self._reset_ratios); bm.addWidget(b_def)
        g.addLayout(bm)
        self._det_lbl = QtWidgets.QLabel("")
        self._det_lbl.setWordWrap(True)
        self._det_lbl.setStyleSheet("font-family:monospace; font-size:10px; color:#4E342E;")
        g.addWidget(self._det_lbl)
        v.addWidget(grp)
        for s_ in (self._s_bf, self._s_in, self._s_out):
            s_.valueChanged.connect(lambda *_: self._update_det_label())
        # compute + save
        self._heavy_buttons = [b_dp, self._b_vac]
        for txt, fn in (("Compute ADF", lambda: self._compute_vi("adf")),
                        ("Compute BF", lambda: self._compute_vi("bf")),
                        ("Compute ADF + BF", lambda: self._compute_vi("both")),
                        ("Save .h5", self._save_h5)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); v.addWidget(b)
            self._heavy_buttons.append(b)
        v.addStretch(1)
        self._dp_progress = QtWidgets.QProgressBar()
        self._dp_progress.setRange(0, 100); self._dp_progress.setValue(0)
        v.addWidget(self._dp_progress)
        self._status = QtWidgets.QLabel("Ready"); self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept); v.addWidget(bb)
        lay.addWidget(panel)

        # live progress: piggy-back on the same tqdm→GUI bridge the main window
        # uses, so this dialog's bar moves during the heavy py4DSTEM passes too.
        self._remove_tqdm_sinks = None
        try:
            import qt_tqdm
        except Exception:
            qt_tqdm = None
        if qt_tqdm is not None:
            def _on_progress(pct: float) -> None:
                try:
                    p = int(max(0, min(100, round(float(pct)))))
                except Exception:
                    return
                self._host.sig_call.emit(lambda p=p: self._dp_progress.setValue(p))
            self._remove_tqdm_sinks = qt_tqdm.add_temp_sinks(progress=_on_progress)
        self._sync_alpha_source()

    # ── vacuum-alpha preset ──────────────────────────────────────────────────
    def _sync_alpha_source(self) -> None:
        """Default to the vacuum alpha when this scan has a vacuum file; if that
        vacuum is already cached, pick it up immediately (cheap)."""
        sc = self._sc
        vac = getattr(sc, "vacuum_path", "") if sc is not None else ""
        self._alpha_src.setCurrentIndex(0 if vac else 1)
        self._b_vac.setEnabled(bool(vac))
        self._update_dp_button()
        if not vac:
            self._update_det_label()
            return
        try:
            cached = E.vacuum_probe_cached(vac)
        except Exception:
            cached = False
        if cached:
            self._load_vacuum_probe()
        else:
            self._update_det_label()

    def _update_dp_button(self) -> None:
        """With the vacuum alpha already known (cache), the button only runs the
        DP pass (center check / display) — no probe work is left to do."""
        if self._alpha_vac is not None:
            self._b_dp.setText("Compute DP (heavy · center check)")
            self._b_dp.setToolTip(
                "alpha is already known from the cached vacuum probe. This only runs\n"
                "the DP mean/max pass to show the pattern and refine the beam center.\n"
                "Optional: you can go straight to Compute ADF/BF.")
        else:
            self._b_dp.setText("Compute DP + probe (heavy)")
            self._b_dp.setToolTip("DP mean/max pass + probe size/center from this scan's DP"
                                  + (" and the vacuum alpha (cached afterwards)."
                                     if getattr(self._sc, "vacuum_path", "") else "."))

    def _load_vacuum_probe(self) -> None:
        sc = self._sc
        vac = getattr(sc, "vacuum_path", "") if sc is not None else ""
        if not vac:
            self._status.setText("This scan has no vacuum file (Choose vacuum file…).")
            return
        if self._host._busy:
            return

        def work():
            return E.vacuum_probe_params(vac, log=self._dialog_log)

        def done(res):
            self._set_dialog_busy(False)
            if not isinstance(res, dict):
                self._status.setText("Vacuum probe failed — see console."); return
            self._alpha_vac = float(res["alpha"])
            self._vac_probe = res.get("probe")
            if self._dp_mean is None and self._dp_max is None:
                # no DP of the scan yet: start from the vacuum center / shape
                self._cy, self._cx = float(res["qx0"]), float(res["qy0"])
                if self._vac_probe is not None and getattr(self._vac_probe, "ndim", 0) == 2:
                    self._qshape = tuple(self._vac_probe.shape)
                    qmax = int(max(self._qshape) // 2)
                    for s_ in (self._s_bf, self._s_in, self._s_out):
                        s_.setMaximum(max(qmax, s_.value()))
            if self._dp_mean is None and self._dp_max is None:
                self._probe_lbl.setText(f"Vacuum: alpha={self._alpha_vac:.4f}  "
                                        f"qx0(row)={res['qx0']:.2f}  qy0(col)={res['qy0']:.2f}")
            self._alpha_src.setCurrentIndex(0)
            self._update_dp_button()
            self._apply_mult()
            self._redraw_dp()
            how = {"memory": "cache (RAM)", "disk": "disk cache",
                   "computed": "vacuum file (now cached)"}.get(res.get("origin"), "?")
            self._status.setText(f"Vacuum alpha = {self._alpha_vac:.2f} px from {how}. "
                                 + ("Check the center on the DP." if self._dp_mean is not None
                                    or self._dp_max is not None else
                                    "Center = vacuum center (compute DP to verify)."))

        self._set_dialog_busy(True)
        self._status.setText("Vacuum probe…")
        self._host._run_async(work, label="Vacuum probe", on_done=done)

    def _current_alpha(self):
        src = self._alpha_src.currentData()
        if src == "vacuum" and self._alpha_vac is not None:
            return self._alpha_vac, "vacuum"
        if self._alpha is not None:
            return self._alpha, "dp"
        if self._alpha_vac is not None:
            return self._alpha_vac, "vacuum"
        return None, None

    def _reset_ratios(self) -> None:
        self._m_bf.setValue(E.DETECTOR_RATIOS["bf"])
        self._m_in.setValue(E.DETECTOR_RATIOS["adf_inner"])
        self._m_out.setValue(E.DETECTOR_RATIOS["adf_outer"])
        self._apply_mult()

    def _on_alpha_mrad(self, val) -> None:
        try:
            QtCore.QSettings("Fast4D", "virtualization").setValue("alpha_mrad", float(val))
        except Exception:
            pass
        self._update_det_label()

    def _update_det_label(self) -> None:
        a, src = self._current_alpha()
        bf, r0, r1 = self._s_bf.value(), self._s_in.value(), self._s_out.value()
        lines = []
        if a is None:
            lines.append("α: —  (Load vacuum probe or Compute DP)")
        else:
            lines.append(f"α = {a:.2f} px ({'vacuum' if src == 'vacuum' else 'scan DP'})")
            if self._alpha is not None and self._alpha_vac is not None:
                lines.append(f"  (vacuum {self._alpha_vac:.2f} | DP {self._alpha:.2f})")
            lines.append(f"BF {bf} px = {bf / a:.2f}α   ADF {r0}-{r1} px = "
                         f"{r0 / a:.2f}-{r1 / a:.2f}α")
            am = float(self._alpha_mrad.value())
            if am > 0:
                k = am / a
                lines.append(f"≈ BF {bf * k:.1f} mrad  HAADF {r0 * k:.1f}-{r1 * k:.1f} mrad")
        if self._det_info.get("clamped"):
            lines.append(f"! outer clamped to the detector edge ({self._det_info['r_edge']:.0f} px)")
        self._det_lbl.setText("\n".join(lines))

    # ── progress / busy-state plumbing ───────────────────────────────────────
    def _dialog_log(self, msg: str) -> None:
        """Same console the host uses, plus mirror the current stage into this
        dialog's own status label (the console sink already runs off-thread —
        the label update is marshaled to the GUI thread like progress is)."""
        self._host._console.log(msg)
        self._host.sig_call.emit(lambda m=msg: self._status.setText(m))

    def _set_dialog_busy(self, busy: bool) -> None:
        for b in self._heavy_buttons:
            b.setEnabled(not busy)
        self._file_combo.setEnabled(not busy)

    def closeEvent(self, ev) -> None:
        if self._remove_tqdm_sinks is not None:
            self._remove_tqdm_sinks()
        super().closeEvent(ev)

    # ── small widget helpers ─────────────────────────────────────────────────
    def _slider(self, layout, label, lo, hi, val):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(f"{label}:"); lbl.setMinimumWidth(110)
        s = _LabeledSlider(lo, hi, 1, val, decimals=0)
        row.addWidget(lbl); row.addWidget(s, 1); layout.addLayout(row)
        return s

    def _dspin(self, lo, hi, val):
        s = QtWidgets.QDoubleSpinBox(); s.setDecimals(2); s.setRange(lo, hi)
        s.setSingleStep(0.1); s.setValue(val); s.setMinimumWidth(80); s.setMaximumWidth(96)
        return s

    def _circle(self, r, n=160):
        t = self._np.linspace(0, 2 * self._np.pi, n)
        return self._cx + r * self._np.cos(t), self._cy + r * self._np.sin(t)

    def _on_file_changed(self, _idx):
        """Switch the target scan (from the loaded files) and reset the DP state."""
        if self._host._busy:
            return
        i = self._file_combo.currentData()
        scans = self._host._scans
        self._sc = scans[i] if (isinstance(i, int) and 0 <= i < len(scans)) else None
        self._dp_mean = self._dp_max = self._alpha = None
        self._alpha_vac = self._vac_probe = self._qshape = None
        self._det_info = {}
        self._drag = None
        try:
            self._img.clear()
        except Exception:
            pass
        for it in (self._c_bf, self._c_in, self._c_out, self._ctr):
            it.setData([], [])
        self._probe_lbl.setText("Probe: (not computed)")
        self._rb_mean.setEnabled(True); self._rb_max.setEnabled(True)
        self._rb_mean.setChecked(True)
        self._dp_progress.setValue(0)
        nm = self._sc.name if self._sc else "?"
        self.setWindowTitle(f"Create ADF / BF / DP  —  {nm}")
        if self._sc is not None and not self._sc.raw_path:
            self._status.setText(f"'{nm}' has no raw 4D path — pick a file with a .mib/.dm4/.h5.")
        else:
            self._status.setText(f"'{nm}' selected. Compute DP+probe to begin.")
        self._sync_alpha_source()

    # ── DP compute / display ─────────────────────────────────────────────────
    def _compute_dp(self):
        if self._sc is None or self._host._busy:
            return
        sc = self._sc
        if not sc.raw_path:
            self._status.setText(f"'{sc.name}' has no raw 4D path to build from."); return
        mode = ("mean" if self._rb_mode_mean.isChecked()
                else "max" if self._rb_mode_max.isChecked() else "both")

        want_vac = bool(getattr(sc, "vacuum_path", "")) and self._alpha_vac is None

        def work():
            res = E.vc_compute_dp_probe(sc, mode=mode, log=self._dialog_log)
            if want_vac:
                try:
                    res["vac"] = E.vacuum_probe_params(sc.vacuum_path, log=self._dialog_log)
                except Exception as exc:
                    self._dialog_log(f"[{sc.name}] vacuum alpha not available: {exc}")
            return res

        def done(res):
            self._set_dialog_busy(False)
            if not isinstance(res, dict):
                self._status.setText("DP/probe failed — see console."); return
            self._dp_mean = res["dp_mean"]; self._dp_max = res["dp_max"]
            self._alpha = res["alpha"]
            self._qshape = res.get("qshape")
            if isinstance(res.get("vac"), dict):
                self._alpha_vac = float(res["vac"]["alpha"])
                self._vac_probe = res["vac"].get("probe")
                self._update_dp_button()
            self._cy, self._cx = float(res["qx0"]), float(res["qy0"])   # (row, col)
            qmax = int(res["qmax"])
            for s in (self._s_bf, self._s_in, self._s_out):
                s.setMaximum(max(qmax, s.value()))
            vtxt = (f"\nVacuum: alpha={self._alpha_vac:.4f}" if self._alpha_vac is not None
                    else "")
            self._probe_lbl.setText(f"DP: alpha={res['alpha']:.4f}  "
                                    f"qx0(row)={res['qx0']:.2f}  qy0(col)={res['qy0']:.2f}"
                                    + vtxt)
            self._rb_mean.setEnabled(self._dp_mean is not None)
            self._rb_max.setEnabled(self._dp_max is not None)
            if self._dp_mean is not None:
                self._rb_mean.setChecked(True)
            elif self._dp_max is not None:
                self._rb_max.setChecked(True)
            self._apply_mult()
            self._redraw_dp()
            self._status.setText("DP+probe ready. Place detectors, then Compute.")

        self._set_dialog_busy(True)
        self._dp_progress.setValue(0)
        self._status.setText("Computing DP (heavy)…" if self._alpha_vac is not None
                             else "Computing DP + probe (heavy)…")
        self._host._run_async(work, label=f"DP+probe ({sc.name})", on_done=done)

    def _redraw_dp(self):
        a = self._dp_mean if self._rb_mean.isChecked() else self._dp_max
        if a is None:
            a = self._dp_max if self._dp_mean is None else self._dp_mean
        if a is None:
            a = self._vac_probe                   # vacuum probe until a DP exists
        if a is None:
            return
        a = self._np.asarray(a, dtype=float)
        self._img.setImage(a)
        v = a[a > 0]
        if v.size:
            lo, hi = float(self._np.percentile(v, 1)), float(self._np.percentile(v, 99.5))
            if not hi > lo:
                lo, hi = 0.0, max(hi, 1e-12)
            self._img.setLevels([lo, hi])
        self._redraw_overlays()
        self._vb.autoRange()

    def _redraw_overlays(self, *_):
        self._c_bf.setData(*self._circle(self._s_bf.value()))
        self._c_in.setData(*self._circle(self._s_in.value()))
        self._c_out.setData(*self._circle(self._s_out.value()))
        self._ctr.setData([self._cx], [self._cy], size=10,
                          pen=self._pg.mkPen("white"), brush=self._pg.mkBrush("red"))

    # ── interactive detector drag (move center / resize a ring edge) ─────────
    def _evt_xy(self, ev):
        pos = ev.position() if hasattr(ev, "position") else ev.pos()
        sp = self._glw.mapToScene(pos.toPoint())
        pt = self._vb.mapSceneToView(sp)
        return float(pt.x()), float(pt.y())

    def _tol(self):
        (x0, x1) = self._vb.viewRange()[0]
        return max(2.0, 0.03 * abs(float(x1) - float(x0)))

    def _pick_mode(self, x, y):
        mode = self._mode_combo.currentData()
        if mode != "auto":
            return mode
        dist = float(self._np.hypot(x - self._cx, y - self._cy))
        cands = sorted([(abs(dist - self._s_bf.value()), "bf"),
                        (abs(dist - self._s_in.value()), "adf_inner"),
                        (abs(dist - self._s_out.value()), "adf_outer")])
        return cands[0][1] if cands[0][0] <= self._tol() else "move"

    def _apply_radius(self, mode, x, y):
        d = int(round(float(self._np.hypot(x - self._cx, y - self._cy))))
        if mode == "bf":
            self._s_bf.setValue(d)
        elif mode == "adf_inner":
            self._s_in.setValue(d)
        elif mode == "adf_outer":
            self._s_out.setValue(d)

    def eventFilter(self, obj, ev):
        if self._dp_mean is None and self._dp_max is None and self._vac_probe is None:
            return False
        t = ev.type()
        T = QtCore.QEvent.Type
        if t == T.MouseButtonPress and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            x, y = self._evt_xy(ev)
            self._drag = self._pick_mode(x, y)
            if self._drag == "move":
                self._cx, self._cy = x, y; self._redraw_overlays()
            elif self._drag:
                self._apply_radius(self._drag, x, y)
            return self._drag is not None
        if t == T.MouseMove and self._drag is not None:
            x, y = self._evt_xy(ev)
            if self._drag == "move":
                self._cx, self._cy = x, y; self._redraw_overlays()
            else:
                self._apply_radius(self._drag, x, y)
            return True
        if t == T.MouseButtonRelease and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._drag is not None:
                self._drag = None
                self._status.setText(f"center (y,x)=({self._cy:.1f}, {self._cx:.1f})  "
                                     f"BF={self._s_bf.value()} ADF=({self._s_in.value()},{self._s_out.value()})")
                self._update_det_label()
                return True
        return False

    def _apply_mult(self):
        a, _src = self._current_alpha()
        if a is None:
            self._update_det_label()
            return
        r = E.detector_radii_from_alpha(
            a, center_yx=(self._cy, self._cx), qshape=self._qshape,
            bf=self._m_bf.value(), adf_inner=self._m_in.value(),
            adf_outer=self._m_out.value(), clamp=self._clamp.isChecked())
        self._det_info = r
        for s_, val in ((self._s_bf, r["bf"]), (self._s_in, r["adf_inner"]),
                        (self._s_out, r["adf_outer"])):
            v_ = int(round(val))
            if v_ > float(getattr(s_, "_hi", v_)):
                s_.setMaximum(v_)
            s_.setValue(v_)
        self._redraw_overlays()
        self._update_det_label()

    # ── virtual images + save ────────────────────────────────────────────────
    def _compute_vi(self, which):
        if self._sc is None or self._host._busy:
            return
        if self._dp_mean is None and self._dp_max is None and self._vac_probe is None:
            self._status.setText("Compute DP+probe (or Load vacuum probe) first."); return
        sc = self._sc
        center = (self._cy, self._cx)                # (row, col)
        adf = (self._s_in.value(), self._s_out.value())
        bf = self._s_bf.value()
        a, src = self._current_alpha()
        meta = {"alpha_px": a, "alpha_source": src,
                "alpha_vacuum_px": self._alpha_vac, "alpha_dp_px": self._alpha,
                "vacuum_path": getattr(sc, "vacuum_path", "") or None,
                "center_source": ("dp" if (self._dp_mean is not None
                                           or self._dp_max is not None) else "vacuum"),
                "ratios_set": [self._m_bf.value(), self._m_in.value(), self._m_out.value()],
                "clamped": bool(self._det_info.get("clamped"))}
        if a:
            meta["ratios_used"] = [bf / a, adf[0] / a, adf[1] / a]
        am = float(self._alpha_mrad.value())
        if a and am > 0:
            k = am / a
            meta.update({"alpha_mrad": am, "bf_mrad": bf * k,
                         "adf_mrad": [adf[0] * k, adf[1] * k]})

        def work():
            if getattr(getattr(sc, "state", None), "datacube", None) is None:
                E.load_datacube(sc, log=self._dialog_log)   # vacuum-only path
            E.vc_compute_virtual_images(sc, which=which, center_yx=center,
                                        adf_radii=adf, bf_radius=bf,
                                        log=self._dialog_log, detector_meta=meta)

        def done(_r):
            self._set_dialog_busy(False)
            key_map = {
                "adf":  [("annular_dark_field", "ADF")],
                "bf":   [("bright_field", "BF")],
                "both": [("annular_dark_field", "ADF"), ("bright_field", "BF")],
            }
            imgs = {}
            for key, label in key_map.get(which, []):
                arr = E.vc_read_virtual(sc, key)
                if arr is not None:
                    imgs[label] = arr
            self._status.setText(f"{which.upper()} computed. Save .h5 to persist "
                                 "(then it becomes the scan's ADF/BF preview).")
            if imgs:
                _show_vimg_preview(self, imgs)

        self._set_dialog_busy(True)
        self._dp_progress.setValue(0)
        self._status.setText(f"Computing {which.upper()} (full 4D pass)…")
        self._host._run_async(work, label=f"Virtual {which} ({sc.name})", on_done=done)

    def _save_h5(self):
        if self._sc is None or self._host._busy:
            return
        sc = self._sc
        from pathlib import Path as _P
        default = ""
        # Prefer the loaded EMD/HDF5 itself so ADF/BF hang on the same file
        # (no new sidecar). Fall back to <raw>.h5 for .mib/.dm4 scans.
        if sc.h5_path and _P(sc.h5_path).suffix.lower() in (".h5", ".hdf5", ".emd"):
            default = sc.h5_path
        elif sc.raw_path:
            rp = _P(sc.raw_path)
            if rp.suffix.lower() in (".h5", ".hdf5", ".emd"):
                default = str(rp)
            else:
                default = str(rp.with_suffix(".h5"))
        st = sc.ensure_state() if hasattr(sc, "ensure_state") else None
        emd_dp = getattr(st, "emd_datapath", None) if st is not None else None
        same_h5 = bool(default) and sc.raw_path and (
            _P(default).resolve() == _P(sc.raw_path).resolve()
            if _P(sc.raw_path).suffix.lower() in (".h5", ".hdf5", ".emd")
            else False
        )
        if same_h5 and emd_dp:
            # Confirm overwrite-append into the open file (preserves sibling groups).
            ans = QtWidgets.QMessageBox.question(
                self,
                "Save virtual images",
                f"Append ADF / BF / DP into the same file?\n\n{default}\n"
                f"(EMD path: {emd_dp})\n\n"
                "Other datasets in this HDF5 (e.g. polyAu, vacuum_probe) are kept.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            p = default
        else:
            p, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save virtual-images .h5", default, "HDF5 (*.h5);;All (*)")
            if not p:
                return

        def work():
            E.vc_save_h5(sc, p, log=self._dialog_log)
            E.load_adf(sc, log=self._dialog_log)   # refresh the ADF preview

        def done(_r):
            self._set_dialog_busy(False)
            self._host._refresh_files(); self._host._update_active_views()
            self._status.setText(f"Saved → {p}")

        self._set_dialog_busy(True)
        self._dp_progress.setValue(0)
        self._status.setText("Saving .h5…")
        self._host._run_async(work, label=f"Save h5 ({sc.name})", on_done=done)


# ─────────────────────────────────────────────────────────────────────────────
# Drift Estimation — measure inter-scan rigid drift via phase cross-correlation
# on strain / ADF maps.  Produces a CSV loadable by engine.load_drift_csv().
# ─────────────────────────────────────────────────────────────────────────────

class DriftEstimateDialog(QtWidgets.QDialog):
    """Inter-scan drift from the ADF / BF of user-chosen scans (reproducibility).

    Use case: the SAME region scanned several times in a row under the same
    conditions (e.g. 3 repeats). Each scan is registered onto a reference by
    phase cross-correlation of its ADF, BF, or both (ADF+BF: measured on each
    and averaged; their difference is reported as the uncertainty). The
    result is a per-file drift CSV (``engine.load_drift_csv``) so the line /
    ROI plugin places the same line on the same physical feature in every file.

    Modes: "vs reference" (every scan against one) or "chained" (each against
    the previous one in the list order, accumulated — for series whose
    contrast changes over time). Order the list by acquisition (drag & drop).

    Quality: NCC before -> after the shift, the ADF/BF disagreement, and an
    overlay (reference magenta, scan green; grey where they match) for the
    selected row. Not intra-scan drift (distortion within one acquisition).
    """

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        self._host = host
        self._scans = host._scans
        self._results: list[dict] = []
        self._run_scans: list = []
        self._run_ref = None
        self._run_key = "adf+bf"
        self.setWindowTitle("Inter-scan drift (ADF / BF)")
        self.resize(1180, 720)
        self._build()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        import drift_estimate as _DE
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        outer = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        outer.addLayout(left, 0)

        info = QtWidgets.QLabel(
            "<b>Same region, scanned several times.</b> Measures how many pixels each "
            "scan moved relative to the reference, from its ADF and/or BF. Tick the "
            "scans to include and drag them into acquisition order.")
        info.setWordWrap(True)
        info.setMaximumWidth(330)
        info.setStyleSheet("background:#E3F2FD; border:1px solid #90CAF9; "
                           "border-radius:4px; padding:7px; color:#0D2A4A;")
        left.addWidget(info)

        left.addWidget(QtWidgets.QLabel("<b>Scans</b> (checked = included; drag to reorder)"))
        self._list = QtWidgets.QListWidget()
        self._list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self._list.setMaximumWidth(330)
        for i, s in enumerate(self._scans):
            it = QtWidgets.QListWidgetItem(s.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                        | QtCore.Qt.ItemFlag.ItemIsDragEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Checked)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            self._list.addItem(it)
        self._list.itemChanged.connect(lambda _it: self._refresh_ref_options())
        self._list.model().rowsMoved.connect(lambda *_: self._refresh_ref_options())
        left.addWidget(self._list, 1)
        row = QtWidgets.QHBoxLayout()
        for txt, st in (("All", True), ("None", False)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(lambda _=False, st=st: self._check_all(st))
            row.addWidget(b)
        left.addLayout(row)

        form = QtWidgets.QFormLayout()
        self._ref_cb = QtWidgets.QComboBox()
        form.addRow("Reference:", self._ref_cb)
        self._chan_cb = QtWidgets.QComboBox()
        for lbl, key in _DE.CHANNELS:
            self._chan_cb.addItem(lbl, key)
        self._chan_cb.setCurrentIndex(0)                 # ADF + BF
        self._chan_cb.setToolTip(
            "ADF + BF: shift measured on each image and averaged; their difference is "
            "reported as the uncertainty (two independent contrasts of the same field).\n"
            "ADF or BF alone, or a strain map if virtual images are not available.")
        form.addRow("Image:", self._chan_cb)
        self._mode_cb = QtWidgets.QComboBox()
        self._mode_cb.addItem("vs reference", "reference")
        self._mode_cb.addItem("chained (each vs previous)", "chained")
        self._mode_cb.setToolTip(
            "vs reference: every scan against the reference (independent errors).\n"
            "chained: each scan against the previous one in the list, accumulated — "
            "use it when contrast changes along the series (contamination, damage); "
            "the error grows with the number of links.")
        form.addRow("Mode:", self._mode_cb)
        self._roi_cb = QtWidgets.QComboBox()
        self._roi_cb.setToolTip(
            "Restrict the correlation to a feature-rich area ROI of the reference "
            "(e.g. an edge). Define it with 'Edit ROIs…' in Set up Lines.")
        form.addRow("Track in ROI:", self._roi_cb)
        self._ups = QtWidgets.QSpinBox(); self._ups.setRange(1, 200); self._ups.setValue(20)
        self._ups.setSuffix("×"); self._ups.setToolTip("Sub-pixel upsampling: 20 → 0.05 px.")
        form.addRow("Precision:", self._ups)
        self._bg = QtWidgets.QDoubleSpinBox(); self._bg.setRange(0.0, 200.0)
        self._bg.setDecimals(1); self._bg.setValue(0.0); self._bg.setSuffix(" px")
        self._bg.setSpecialValueText("auto")
        self._bg.setToolTip("Gaussian width of the smooth background removed before "
                            "correlating (thickness / illumination gradients). 0 = auto.")
        form.addRow("Background:", self._bg)
        self._win = QtWidgets.QCheckBox("taper edges (Tukey window)")
        self._win.setChecked(True)
        form.addRow("", self._win)
        left.addLayout(form)
        self._refresh_ref_options()
        self._ref_cb.currentIndexChanged.connect(self._refresh_roi_options)

        self._btn_run = QtWidgets.QPushButton("▶  Estimate drift")
        self._btn_run.setStyleSheet(
            "QPushButton{background:#1565C0; color:white; font-weight:bold; "
            "border-radius:5px; padding:6px 16px;}"
            "QPushButton:hover{background:#1976D2;}"
            "QPushButton:disabled{background:#90CAF9; color:#ccc;}")
        self._btn_run.clicked.connect(self._run)
        left.addWidget(self._btn_run)

        # right: results, summary, preview, actions
        right = QtWidgets.QVBoxLayout()
        outer.addLayout(right, 1)
        self._table = QtWidgets.QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["Scan", "dy (px)", "dx (px)", "|shift| px", "|shift| nm",
             "NCC before → after", "ADF/BF diff (px)", "Notes"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._preview)
        right.addWidget(self._table, 2)
        self._summary = QtWidgets.QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color:#0D2A4A;")
        right.addWidget(self._summary)

        # two views: every selected scan overlaid (before | after the correction),
        # and the reference against the scan picked in the table
        self._tabs = QtWidgets.QTabWidget()
        self._ofig = Figure(figsize=(8, 3.6))
        self._ocanvas = FigureCanvasQTAgg(self._ofig)
        self._tabs.addTab(self._ocanvas, "All scans overlaid: before | after")
        self._fig = Figure(figsize=(8, 2.8))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._tabs.addTab(self._canvas, "Selected scan vs reference")
        self._tabs.setMinimumHeight(260)
        right.addWidget(self._tabs, 3)
        self._overlays = None

        act = QtWidgets.QHBoxLayout()
        self._status_lbl = QtWidgets.QLabel("")
        self._status_lbl.setStyleSheet("color:#1565C0; font-size:11px;")
        act.addWidget(self._status_lbl, 1)
        self._btn_save = QtWidgets.QPushButton("Save drift CSV…")
        self._btn_save.setEnabled(False)
        self._btn_save.setToolTip("stem, shift_dy_px, shift_dx_px (+ QC columns). "
                                  "Reload it with 'Load drift CSV' in the line / ROI tools.")
        self._btn_save.clicked.connect(self._save_csv)
        self._btn_load = QtWidgets.QPushButton("Load into session")
        self._btn_load.setEnabled(False)
        self._btn_load.setToolTip("Assign the shifts to these scans now; then use "
                                  "'Propagate (with drift)' in Set up Lines.")
        self._btn_load.clicked.connect(self._load_into_session)
        self._btn_ovl = QtWidgets.QPushButton("Save overlay images…")
        self._btn_ovl.setEnabled(False)
        self._btn_ovl.setToolTip("Two PNGs with a colour legend: all scans overlaid as "
                                 "acquired (drift visible) and after the correction.")
        self._btn_ovl.clicked.connect(self._save_overlays)
        act.addWidget(self._btn_ovl)
        act.addWidget(self._btn_save)
        act.addWidget(self._btn_load)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        act.addWidget(bb)
        right.addLayout(act)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _check_all(self, on: bool) -> None:
        st = QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(st)

    def _selected_scans(self) -> list:
        out = []
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.checkState() == QtCore.Qt.CheckState.Checked:
                j = it.data(QtCore.Qt.ItemDataRole.UserRole)
                if 0 <= j < len(self._scans):
                    out.append(self._scans[j])
        return out

    def _refresh_ref_options(self) -> None:
        cur = self._ref_cb.currentText()
        sel = self._selected_scans()
        self._ref_cb.blockSignals(True)
        self._ref_cb.clear()
        for s in sel:
            self._ref_cb.addItem(s.name)
        k = self._ref_cb.findText(cur)
        if k < 0 and 0 <= self._host._active < len(self._scans):
            k = self._ref_cb.findText(self._scans[self._host._active].name)
        self._ref_cb.setCurrentIndex(max(k, 0))
        self._ref_cb.blockSignals(False)
        self._refresh_roi_options()

    def _ref_scan(self):
        name = self._ref_cb.currentText()
        return next((s for s in self._scans if s.name == name), None)

    def _refresh_roi_options(self, *_) -> None:
        self._roi_cb.clear()
        self._roi_cb.addItem("(whole map)", None)
        ref = self._ref_scan()
        rois = E.scan_area_rois(ref) if ref is not None else {}
        for rid, bounds in rois.items():
            self._roi_cb.addItem(f"{rid}  {[int(b) for b in bounds]}", list(bounds))

    @staticmethod
    def _px_nm(scan) -> float | None:
        try:
            pm = float(getattr(scan.params, "r_px_pm", 0.0) or 0.0)
        except Exception:
            pm = 0.0
        return pm / 1000.0 if pm > 0 else None

    # ── actions ───────────────────────────────────────────────────────────────
    def _run(self) -> None:
        scans = self._selected_scans()
        ref = self._ref_scan()
        if len(scans) < 2 or ref is None:
            QtWidgets.QMessageBox.information(
                self, "Inter-scan drift",
                "Tick at least 2 scans of the same region (and pick the reference).")
            return
        key = self._chan_cb.currentData()
        mode = self._mode_cb.currentData()
        kw = dict(image_key=key, mode=mode, upsample_factor=self._ups.value(),
                  roi_bounds=self._roi_cb.currentData(), background_sigma=self._bg.value(),
                  window=self._win.isChecked(), log=self._host._console.log)
        for b in (self._btn_run, self._btn_save, self._btn_load, self._btn_ovl):
            b.setEnabled(False)
        self._status_lbl.setText("Loading ADF/BF and cross-correlating…")
        self._table.setRowCount(0)
        import drift_estimate as _DE

        def work():
            for s in scans:                      # make sure the ADF/BF previews are loaded
                if E.cached_adf(s) is None:
                    try:
                        E.load_adf(s, log=self._host._console.log)
                    except Exception:
                        pass
            res = _DE.estimate_drift(scans, ref, **kw)
            try:
                ovl = _DE.overlay_maps(scans, res, key)
            except Exception as exc:
                self._host._console.log(f"[Drift] overlay skipped: {exc}")
                ovl = None
            return res, ovl

        def done(results):
            self._btn_run.setEnabled(True)
            if results is None:
                self._status_lbl.setText("Error — see console.")
                return
            results, self._overlays = results
            self._results, self._run_scans, self._run_ref, self._run_key = results, scans, ref, key
            self._populate_table(results)
            self._draw_overlays()
            self._btn_save.setEnabled(True)
            self._btn_load.setEnabled(True)
            self._btn_ovl.setEnabled(self._overlays is not None)
            self._status_lbl.setText(f"Done — {len(results)} scan(s).")
            if len(results) > 1:
                self._table.selectRow(1 if results[0].get("is_reference") else 0)

        self._host._run_async(work, label="Estimate drift", on_done=done)

    def _populate_table(self, results: list[dict]) -> None:
        import math
        import drift_estimate as _DE
        px = self._px_nm(self._run_ref) if self._run_ref is not None else None
        self._table.setRowCount(len(results))

        def item(text, bold=False, color=None):
            it = QtWidgets.QTableWidgetItem(text)
            it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if bold:
                f = it.font(); f.setBold(True); it.setFont(f)
            if color:
                it.setForeground(QtGui.QColor(color))
            return it

        fin = lambda v: v is not None and not (isinstance(v, float) and math.isnan(v))
        for r_, r in enumerate(results):
            self._table.setItem(r_, 0, item(r["name"], bold=r["is_reference"]))
            self._table.setItem(r_, 1, item(f"{r['dy']:+.2f}"))
            self._table.setItem(r_, 2, item(f"{r['dx']:+.2f}"))
            self._table.setItem(r_, 3, item(f"{r['magnitude']:.2f}"))
            self._table.setItem(r_, 4, item(f"{r['magnitude'] * px:.2f}" if px else "—"))
            nb, na = r.get("ncc_before"), r.get("ncc_after")
            col = "#C62828" if fin(na) and na < 0.5 and not r["is_reference"] else None
            self._table.setItem(r_, 5, item("—" if r["is_reference"] else
                                            f"{nb:.2f} → {na:.2f}" if fin(na) else "—", color=col))
            d = r.get("adf_bf_diff_px")
            self._table.setItem(r_, 6, item(f"{d:.2f}" if fin(d) and not r["is_reference"] else "—",
                                            color="#E65100" if fin(d) and d > 1.0 else None))
            note = "REFERENCE" if r["is_reference"] else r.get("warning", "")
            self._table.setItem(r_, 7, item(note, color="#1565C0" if r["is_reference"]
                                            else ("#E65100" if note else None)))
        self._table.resizeColumnsToContents()
        s = _DE.summary(results, px_nm=px)
        unit = (lambda v_px, v_nm: f"{v_px:.2f} px ({v_nm:.2f} nm)") if px else \
               (lambda v_px, v_nm: f"{v_px:.2f} px")
        self._summary.setText(
            f"<b>{s['n']} scans</b> · largest shift from the reference "
            f"{unit(s['max_shift_px'], s.get('max_shift_nm', 0))} · mean step between "
            f"consecutive scans {unit(s['mean_step_px'], s.get('mean_step_nm', 0))} · "
            f"worst NCC after alignment {s['ncc_after_min']:.2f}"
            + ("" if px else "  <i>(set 'R px' in the Stress tab for nm)</i>"))

    def _preview(self) -> None:
        """Reference vs selected scan: overlay before / after the shift."""
        import numpy as np
        import drift_estimate as _DE
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if not rows or not self._results:
            return
        r = self._results[rows[0].row()]
        sc = next((s for s in self._run_scans if s.name == r["name"]), None)
        if sc is None or self._run_ref is None:
            return
        key = "adf" if self._run_key == "adf+bf" else self._run_key
        a = _DE._get_map(self._run_ref, key)
        b = _DE._get_map(sc, key)
        if (a is None or b is None) and self._run_key == "adf+bf":
            key = "bf"
            a, b = _DE._get_map(self._run_ref, key), _DE._get_map(sc, key)
        if a is None or b is None:
            return
        if a.shape != b.shape:
            h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]

        def norm(x):
            x = np.asarray(x, float)
            lo, hi = np.nanpercentile(x, (1, 99))
            return np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)

        A = norm(a)
        B0 = norm(b)
        B1 = norm(_DE.shift_image(b, r["dy"], r["dx"]))
        B1 = np.where(np.isfinite(B1), B1, 0.0)

        def rgb(p, q):          # reference magenta, scan green: grey where they agree
            return np.dstack([p, q, p])

        self._fig.clear()
        axs = self._fig.subplots(1, 3)
        axs[0].imshow(A, cmap="gray"); axs[0].set_title(f"reference {self._run_ref.name}", fontsize=9)
        axs[1].imshow(rgb(A, B0)); axs[1].set_title(f"{sc.name} as acquired", fontsize=9)
        axs[2].imshow(rgb(A, B1))
        axs[2].set_title(f"after shift ({r['dy']:+.2f}, {r['dx']:+.2f}) px", fontsize=9)
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        bounds = self._roi_cb.currentData()
        if bounds:
            from matplotlib.patches import Rectangle
            x0, x1, y0, y1 = bounds
            for ax in axs:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=0.8))
        self._fig.text(0.5, 0.01, f"{key.upper()} · magenta = reference, green = scan, "
                                  "grey/white = overlap", ha="center", fontsize=8)
        self._fig.tight_layout(rect=(0, 0.05, 1, 1))
        self._canvas.draw_idle()

    def _draw_overlays(self) -> None:
        """All selected scans in one image each, before and after the correction."""
        import drift_estimate as _DE
        from matplotlib.patches import Patch
        self._ofig.clear()
        if self._overlays is None:
            self._ocanvas.draw_idle()
            return
        before, after, ch, names = self._overlays
        axs = self._ofig.subplots(1, 2)
        axs[0].imshow(before, interpolation="nearest")
        axs[0].set_title("as acquired — coloured fringes = drift", fontsize=9)
        axs[1].imshow(after, interpolation="nearest")
        axs[1].set_title("after correction — grey/white = scans agree", fontsize=9)
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        cols = _DE.overlay_colors(len(names))
        ref = getattr(self._run_ref, "name", "")
        self._ofig.legend(handles=[Patch(facecolor=c, edgecolor="k",
                                         label=n + ("  (ref)" if n == ref else ""))
                                   for n, c in zip(names, cols)],
                          loc="lower center", ncol=min(len(names), 6), fontsize=8, frameon=False)
        self._ofig.text(0.01, 0.99, ch.upper(), va="top", fontsize=8)
        self._ofig.tight_layout(rect=(0, 0.1, 1, 0.97))
        self._ocanvas.draw_idle()
        self._tabs.setCurrentIndex(0)

    def _write_overlays(self, folder, stem) -> list:
        import drift_estimate as _DE
        before, after, ch, names = self._overlays
        return _DE.save_overlays(before, after, names, folder, stem=stem, channel=ch,
                                 results=self._results, px_nm=self._px_nm(self._run_ref))

    def _save_overlays(self) -> None:
        if self._overlays is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save overlay images (…_before.png / …_after.png)",
            "drift_overlay.png", "PNG (*.png)")
        if not path:
            return
        from pathlib import Path as _P
        p = _P(path)
        out = self._write_overlays(p.parent, p.stem)
        self._status_lbl.setText("Saved " + ", ".join(o.name for o in out))
        self._host._console.log("[Drift] overlays saved: " + ", ".join(str(o) for o in out))

    def _save_csv(self) -> None:
        if not self._results:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save drift CSV", "drift_shifts.csv", "CSV (*.csv);;All (*)")
        if not path:
            return
        import drift_estimate as _DE
        saved = _DE.save_drift_csv(self._results, path,
                                   reference=getattr(self._run_ref, "name", None))
        msg = f"Saved → {saved.name}"
        if self._overlays is not None:          # the evidence travels with the numbers
            try:
                out = self._write_overlays(saved.parent, saved.stem + "_overlay")
                msg += " + " + ", ".join(o.name for o in out)
            except Exception as exc:
                self._host._console.log(f"[Drift] overlay images not saved: {exc}")
        self._status_lbl.setText(msg)
        self._host._console.log(f"[Drift] {msg} in {saved.parent}")

    def _load_into_session(self) -> None:
        if not self._results:
            return
        n = 0
        for r in self._results:
            for sc in self._run_scans:
                if sc.name == r["name"]:
                    sc.drift = (float(r["dx"]), float(r["dy"]))
                    n += 1
        self._host._console.log(
            f"[Drift] shifts loaded into session for {n} scan(s). Use 'Propagate (with "
            "drift)' in Set up Lines, or 'Apply drift' in the multi-file line tool.")
        self._status_lbl.setText(f"Shifts applied to {n} scan(s).")



class ReproducibilityDialog(QtWidgets.QDialog):
    """Reproducibility experiment: the same region scanned N times in a row with
    nothing changed on the microscope. Every metric is shown WITHOUT and WITH
    the inter-scan drift correction (the drift loaded from the drift plugin),
    so the drift part of the apparent irreproducibility is separated from the
    measurement's own scatter. See reproducibility.py for the definitions."""

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        self._host = host
        self._scans = host._scans
        self._res = None
        self._run_scans: list = []
        self.setWindowTitle("Reproducibility experiment")
        self.resize(1320, 860)
        self._build()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        self._FC, self._TB, self._Fig = FigureCanvasQTAgg, NavigationToolbar2QT, Figure
        outer = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        info = QtWidgets.QLabel(
            "<b>Same region, scanned consecutively, nothing changed.</b> Tick the scans, "
            "order them by acquisition (drag), pick the reference. Lines and ROIs are "
            "taken from the reference scan; every metric is computed without and with "
            "drift correction.")
        info.setWordWrap(True); info.setMaximumWidth(320)
        info.setStyleSheet("background:#E3F2FD; border:1px solid #90CAF9; border-radius:4px; "
                           "padding:7px; color:#0D2A4A;")
        left.addWidget(info)
        self._list = QtWidgets.QListWidget()
        self._list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self._list.setMaximumWidth(320)
        for i, s in enumerate(self._scans):
            it = QtWidgets.QListWidgetItem(s.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                        | QtCore.Qt.ItemFlag.ItemIsDragEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Checked)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            self._list.addItem(it)
        self._list.itemChanged.connect(lambda _it: self._refresh_ref())
        left.addWidget(self._list, 1)
        form = QtWidgets.QFormLayout()
        self._ref_cb = QtWidgets.QComboBox()
        self._ref_cb.currentIndexChanged.connect(lambda _i: self._refresh_status())
        form.addRow("Reference:", self._ref_cb)
        self._label_cb = QtWidgets.QComboBox()
        self._label_cb.addItem("without ROI (global reference)", "without_roi")
        self._label_cb.addItem("with ROI (experimental reference)", "with_roi")
        self._label_cb.setToolTip("Which strain maps: with a common global reference, or "
                                  "each zeroed on its own ROI (removes offsets between scans).")
        form.addRow("Strain maps:", self._label_cb)
        self._width = QtWidgets.QSpinBox(); self._width.setRange(1, 51); self._width.setValue(3)
        self._width.setSuffix(" px")
        self._width.setToolTip("Line integration width (same physical width on every scan).")
        form.addRow("Line width:", self._width)
        self._imgs = QtWidgets.QCheckBox("include ADF / BF profiles and maps")
        self._imgs.setChecked(True)
        form.addRow("", self._imgs)
        left.addLayout(form)
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True); self._status.setMaximumWidth(320)
        left.addWidget(self._status)
        b_drift = QtWidgets.QPushButton("Estimate drift (ADF / BF)…")
        b_drift.clicked.connect(lambda: self._host._show_tool(DriftEstimateDialog(self._host)))
        left.addWidget(b_drift)
        b_ref = QtWidgets.QPushButton("Refresh lines / drift status")
        b_ref.clicked.connect(self._refresh_status)
        left.addWidget(b_ref)
        self._btn_run = QtWidgets.QPushButton("▶  Compute reproducibility")
        self._btn_run.setStyleSheet(
            "QPushButton{background:#1565C0; color:white; font-weight:bold; "
            "border-radius:5px; padding:6px 16px;}"
            "QPushButton:disabled{background:#90CAF9; color:#ccc;}")
        self._btn_run.clicked.connect(self._run)
        left.addWidget(self._btn_run)
        outer.addLayout(left, 0)

        right = QtWidgets.QVBoxLayout()
        self._tabs = QtWidgets.QTabWidget()
        self._tbl_sum = self._table()
        self._tabs.addTab(self._tbl_sum, "Summary (no drift | drift corrected)")
        # lines tab
        wl = QtWidgets.QWidget(); vl = QtWidgets.QVBoxLayout(wl)
        rl = QtWidgets.QHBoxLayout(); rl.addWidget(QtWidgets.QLabel("Line:"))
        self._line_cb = QtWidgets.QComboBox(); self._line_cb.currentIndexChanged.connect(self._show_line)
        rl.addWidget(self._line_cb); rl.addStretch(1); vl.addLayout(rl)
        self._line_host = QtWidgets.QVBoxLayout(); vl.addLayout(self._line_host, 3)
        self._tbl_line = self._table(); vl.addWidget(self._tbl_line, 1)
        self._tabs.addTab(wl, "Lines")
        # maps tab
        wm = QtWidgets.QWidget(); vm = QtWidgets.QVBoxLayout(wm)
        rm = QtWidgets.QHBoxLayout(); rm.addWidget(QtWidgets.QLabel("Map:"))
        self._map_cb = QtWidgets.QComboBox(); self._map_cb.currentIndexChanged.connect(self._show_map)
        rm.addWidget(self._map_cb); rm.addStretch(1); vm.addLayout(rm)
        self._map_host = QtWidgets.QVBoxLayout(); vm.addLayout(self._map_host, 3)
        self._tbl_map = self._table(); vm.addWidget(self._tbl_map, 1)
        self._tabs.addTab(wm, "Maps: offset / RMSE / s(r)")
        # ROIs tab
        self._tbl_roi = self._table()
        self._tabs.addTab(self._tbl_roi, "ROIs and contrast")
        # drift tab
        wd = QtWidgets.QWidget(); vd = QtWidgets.QHBoxLayout(wd)
        self._drift_host = QtWidgets.QVBoxLayout(); vd.addLayout(self._drift_host, 1)
        rd = QtWidgets.QVBoxLayout(); self._tbl_drift = self._table(); rd.addWidget(self._tbl_drift, 2)
        self._tbl_q = self._table(); rd.addWidget(QtWidgets.QLabel("<b>Fit quality</b>")); rd.addWidget(self._tbl_q, 1)
        vd.addLayout(rd, 1)
        self._tabs.addTab(wd, "Drift and fit quality")
        # per-scan distributions and PCA (moved here from Report > Session analysis)
        wb = QtWidgets.QWidget(); self._box_host = QtWidgets.QVBoxLayout(wb)
        self._tabs.addTab(wb, "Distributions (box / violin)")
        wp = QtWidgets.QWidget(); self._pca_host = QtWidgets.QVBoxLayout(wp)
        self._tabs.addTab(wp, "PCA of scans")
        right.addWidget(self._tabs, 1)
        self._notes = QtWidgets.QLabel(""); self._notes.setWordWrap(True)
        self._notes.setStyleSheet("color:#8a4b00;")
        right.addWidget(self._notes)
        act = QtWidgets.QHBoxLayout(); act.addStretch(1)
        self._btn_exp = QtWidgets.QPushButton("Export report (CSV + PNG)…")
        self._btn_exp.clicked.connect(self._export); self._btn_exp.setEnabled(False)
        self._btn_rep = QtWidgets.QPushButton("Send figures to Report")
        self._btn_rep.clicked.connect(self._send_report); self._btn_rep.setEnabled(False)
        act.addWidget(self._btn_exp); act.addWidget(self._btn_rep)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept); act.addWidget(bb)
        right.addLayout(act)
        outer.addLayout(right, 1)
        self._refresh_ref()

    def _table(self):
        t = QtWidgets.QTableWidget(0, 0)
        t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        sc = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Copy, t)
        sc.activated.connect(lambda t=t: self._copy_table(t))
        return t

    @staticmethod
    def _copy_table(t):
        rows = sorted({i.row() for i in t.selectedIndexes()}) or range(t.rowCount())
        head = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
        lines = ["\t".join(head)]
        for r in rows:
            lines.append("\t".join((t.item(r, c).text() if t.item(r, c) else "")
                                   for c in range(t.columnCount())))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    @staticmethod
    def _fill(t, df, highlight=None):
        import math
        t.clear()
        if df is None or len(df) == 0:
            t.setRowCount(0); t.setColumnCount(0); return
        cols = list(df.columns)
        t.setColumnCount(len(cols)); t.setHorizontalHeaderLabels(cols); t.setRowCount(len(df))
        for r, row in enumerate(df.itertuples(index=False)):
            for c, v in enumerate(row):
                if isinstance(v, float):
                    txt = "—" if math.isnan(v) else (f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}")
                else:
                    txt = str(v)
                it = QtWidgets.QTableWidgetItem(txt)
                if highlight and cols[c] in highlight:
                    it.setBackground(QtGui.QColor(highlight[cols[c]]))
                t.setItem(r, c, it)
        t.resizeColumnsToContents()

    def _set_fig(self, host, fig):
        while host.count():
            w = host.takeAt(0).widget()
            if w is not None:
                w.setParent(None); w.deleteLater()
        canvas = self._FC(fig)
        host.addWidget(self._TB(canvas, self)); host.addWidget(canvas, 1)
        canvas.draw_idle()

    # ── state ─────────────────────────────────────────────────────────────────
    def _selected(self):
        out = []
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(self._scans[it.data(QtCore.Qt.ItemDataRole.UserRole)])
        return out

    def _refresh_ref(self):
        cur = self._ref_cb.currentText()
        self._ref_cb.blockSignals(True); self._ref_cb.clear()
        for s in self._selected():
            self._ref_cb.addItem(s.name)
        k = self._ref_cb.findText(cur)
        if k < 0 and 0 <= self._host._active < len(self._scans):
            k = self._ref_cb.findText(self._scans[self._host._active].name)
        self._ref_cb.setCurrentIndex(max(k, 0)); self._ref_cb.blockSignals(False)
        self._refresh_status()

    def _ref(self):
        return next((s for s in self._scans if s.name == self._ref_cb.currentText()), None)

    def _refresh_status(self):
        sel, ref = self._selected(), self._ref()
        nd = sum(1 for s in sel if getattr(s, "drift", None))
        lines = sorted((getattr(ref, "lines", None) or {})) if ref else []
        rois = sorted(E.scan_area_rois(ref)) if ref else []
        txt = [f"<b>{len(sel)}</b> scans selected",
               f"lines on the reference: <b>{', '.join(lines) or 'none'}</b>",
               f"area ROIs on the reference: <b>{', '.join(rois) or 'none'}</b>",
               (f"drift loaded on <b>{nd}/{len(sel)}</b> scans" if nd else
                "<span style='color:#C62828'><b>no drift loaded</b> — estimate it or load "
                "the drift CSV; otherwise both columns are identical</span>")]
        self._status.setText("<br>".join(txt))

    # ── actions ───────────────────────────────────────────────────────────────
    def _run(self):
        scans, ref = self._selected(), self._ref()
        if len(scans) < 2 or ref is None:
            QtWidgets.QMessageBox.information(self, "Reproducibility",
                                              "Tick at least 2 scans and pick the reference.")
            return
        label = self._label_cb.currentData()
        width = self._width.value()
        imgs = ("adf", "bf") if self._imgs.isChecked() else ()
        self._btn_run.setEnabled(False)
        self._notes.setText("Computing…")
        import reproducibility as RP

        def work():
            for s in scans:
                if imgs and E.cached_adf(s) is None:
                    try:
                        E.load_adf(s, log=self._host._console.log)
                    except Exception:
                        pass
            return RP.run(scans, ref, label=label, width=width, images=imgs,
                          log=self._host._console.log)

        def done(res):
            self._btn_run.setEnabled(True)
            if res is None:
                self._notes.setText("Error — see console."); return
            self._res, self._run_scans = res, scans
            self._show_all()

        self._host._run_async(work, label="Reproducibility", on_done=done)

    def _show_all(self):
        import reproducibility as RP
        res = self._res
        hl = {"no_drift": "#FDECEA", "drift": "#E3F2FD"}
        self._fill(self._tbl_sum, RP.summary_table(res), hl)
        self._line_cb.blockSignals(True); self._line_cb.clear()
        self._line_cb.addItems(list(res.lines)); self._line_cb.blockSignals(False)
        self._map_cb.blockSignals(True); self._map_cb.clear()
        for ch in res.maps:
            self._map_cb.addItem(RP.LABEL.get(ch, ch), ch)
        self._map_cb.blockSignals(False)
        self._show_line(); self._show_map()
        import pandas as pd
        rows = []
        for rid, chs in res.rois.items():
            for ch, m in chs.items():
                for mode in RP.MODES:
                    s = m[mode]["summary"]
                    rows.append(dict(object=rid, channel=ch, mode=mode, mean=s["mean"],
                                     std=s["std"], cv_pct=s["cv_pct"],
                                     **{n: v for n, v in zip(res.names, m[mode]["values"])}))
        for key, chs in res.contrast.items():
            for ch, m in chs.items():
                for mode in RP.MODES:
                    s = m[mode]["summary"]
                    rows.append(dict(object=f"contrast {key}", channel=ch, mode=mode, mean=s["mean"],
                                     std=s["std"], cv_pct=s["cv_pct"],
                                     **{n: v for n, v in zip(res.names, m[mode]["values"])}))
        self._fill(self._tbl_roi, pd.DataFrame(rows))
        self._fill(self._tbl_drift, RP.drift_table(res))
        self._fill(self._tbl_q, RP.quality_table(res))
        self._set_fig(self._drift_host, RP.drift_figure(res))
        self._show_scan_views(res.label)
        self._notes.setText("  ".join(res.notes))
        self._btn_exp.setEnabled(True); self._btn_rep.setEnabled(True)

    def _show_scan_views(self, label):
        """Box/violin by channel and PCA of the scans used in the run."""
        import analysis as A
        for host, make in ((self._box_host,
                            lambda: A.boxplot_figure(self._run_scans, label=label, kind="violin")),
                           (self._pca_host, lambda: A.pca_figure(self._run_scans, label))):
            try:
                self._set_fig(host, make())
            except Exception as exc:
                self._host._console.log(f"[Repro] {exc}")

    def _show_line(self, *_):
        import pandas as pd
        import reproducibility as RP
        if self._res is None or not self._line_cb.currentText():
            return
        lid = self._line_cb.currentText()
        self._set_fig(self._line_host, RP.line_figure(self._res, lid))
        rows = []
        for ch, m in self._res.lines[lid].items():
            for mode in RP.MODES:
                for p in m[mode]["per_scan"]:
                    rows.append(dict(channel=ch, mode=mode, **p))
        self._fill(self._tbl_line, pd.DataFrame(rows))

    def _show_map(self, *_):
        import pandas as pd
        import reproducibility as RP
        ch = self._map_cb.currentData()
        if self._res is None or not ch:
            return
        self._set_fig(self._map_host, RP.map_figure(self._res, ch))
        rows = [dict(mode=mode, **p) for mode in RP.MODES
                for p in self._res.maps[ch][mode]["per_scan"]]
        self._fill(self._tbl_map, pd.DataFrame(rows))

    def _export(self):
        if self._res is None:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the reproducibility report")
        if not d:
            return
        import reproducibility as RP
        out = RP.export(self._res, d)
        self._host._console.log(f"[Repro] report exported: {len(out)} files in {d}")
        self._notes.setText(f"Exported {len(out)} files to {d}")

    def _send_report(self):
        """Register the figures on the reference scan (Report → Reports)."""
        import reproducibility as RP
        res = self._res
        ref = next((s for s in self._run_scans if s.name == res.ref), None)
        if ref is None:
            return
        n = 0
        for lid in res.lines:
            n += bool(E.register_figure(ref, f"report_repro_line_{lid}_{res.label}",
                                        RP.line_figure(res, lid), force=True))
        for ch in res.maps:
            n += bool(E.register_figure(ref, f"report_repro_map_{ch}_{res.label}",
                                        RP.map_figure(res, ch), force=True))
        n += bool(E.register_figure(ref, f"report_repro_drift_{res.label}", RP.drift_figure(res),
                                    force=True))
        try:
            self._host._update_active_views()
            self._host._params.report_refresh()
            report = self._host._params.report
            keys = [k for k in E.list_figure_keys(ref) if k.startswith("report_repro_")]
            if report is not None and keys:
                report.select_figure(self._host._scans.index(ref), keys[0])
                self._host._params._tabs.setCurrentWidget(report)
        except Exception:
            pass
        self._host._console.log(f"[Repro] {n} figure(s) sent to the Report under {ref.name}.")
        self._notes.setText(f"{n} figure(s) added to the Report ({ref.name} → Reports).")

# ─────────────────────────────────────────────────────────────────────────────
# Area ROI editor (the live Line / ROI profiles tool lives in qt_measure.py)
# ─────────────────────────────────────────────────────────────────────────────

class AreaRoiEditorDialog(QtWidgets.QDialog):
    """Edit ALL area ROIs in a single window — no per-ROI OK.

    The active scan's ADF is shown with one draggable / resizable rectangle per
    ROI.  A side table lists every ROI with editable x0/x1/y0/y1 cells; Add,
    Duplicate and Delete operate on the selected row.  Dragging a rectangle
    updates its table row and vice-versa (live, two-way).  «Apply» writes the
    whole set to the chosen files at once.
    """

    _ROI_COLORS = ["#00E5FF", "#FF6D00", "#76FF03", "#E040FB", "#FFEA00", "#18FFFF",
                   "#FF4081", "#B2FF59", "#40C4FF", "#FFD740"]

    def __init__(self, host) -> None:
        super().__init__(host)
        _enable_minmax(self)
        import numpy as np
        import pyqtgraph as pg
        self._np, self._pg = np, pg
        self._host = host
        self._scans = host._scans
        self._scan = host.active_scan() or (self._scans[0] if self._scans else None)
        self._entries: list[dict] = []     # {id, roi(pg.RectROI)}
        self._syncing = False              # guard against table<->rect recursion
        self.setWindowTitle("Area ROI editor")
        self.resize(1100, 680)
        self._build()
        self._load_existing()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        pg = self._pg
        lay = QtWidgets.QHBoxLayout(self)

        # left: ADF image with ROI rectangles
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._glw = pg.GraphicsLayoutWidget()
        self._vb = self._glw.addViewBox()
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)
        self._img = pg.ImageItem()
        self._vb.addItem(self._img)
        lay.addWidget(self._glw, 1)

        # right: control panel
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(360)
        v = QtWidgets.QVBoxLayout(panel)

        v.addWidget(QtWidgets.QLabel("<b>Area ROIs</b>"))
        # file selector (which scan's ADF + ROIs to edit)
        frow = QtWidgets.QHBoxLayout()
        frow.addWidget(QtWidgets.QLabel("File:"))
        self._file_cb = QtWidgets.QComboBox()
        for s in self._scans:
            self._file_cb.addItem(s.name)
        if self._scan is not None and self._scan in self._scans:
            self._file_cb.setCurrentIndex(self._scans.index(self._scan))
        self._file_cb.currentIndexChanged.connect(self._on_file_changed)
        frow.addWidget(self._file_cb, 1)
        v.addLayout(frow)

        # ROI table — editable bounds. Columns auto-fit the panel width so the
        # table never overflows the window (no horizontal scroll needed).
        self._table = QtWidgets.QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["ROI", "x0", "x1", "y0", "y1"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._table.itemChanged.connect(self._on_cell_edited)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        v.addWidget(self._table, 1)

        # row of edit buttons
        btns = QtWidgets.QHBoxLayout()
        for txt, fn, tip in (
            ("+ Add", self._add_roi, "Add a new ROI rectangle at the center."),
            ("Duplicate", self._duplicate_roi, "Copy the selected ROI (offset slightly)."),
            ("Delete", self._delete_roi, "Delete the selected ROI."),
        ):
            b = QtWidgets.QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            btns.addWidget(b)
        v.addLayout(btns)

        # apply scope
        v.addWidget(QtWidgets.QLabel("<b>Apply to</b>"))
        self._apply_all = QtWidgets.QCheckBox(
            "All loaded files (else only this file)")
        self._apply_all.setToolTip(
            "Checked: the ROI set is propagated to every loaded file "
            "(shifted per-file drift when available).\n"
            "Unchecked: ROIs are saved only to the file selected above.")
        v.addWidget(self._apply_all)
        self._use_drift = QtWidgets.QCheckBox("Use per-file drift when applying to all")
        self._use_drift.setChecked(
            any(getattr(s, "drift", None) for s in self._scans))
        v.addWidget(self._use_drift)

        b_apply = QtWidgets.QPushButton("Apply ROIs")
        b_apply.setStyleSheet(
            "QPushButton{background:#1565C0; color:white; font-weight:bold;"
            "border-radius:5px; padding:5px 16px;}"
            "QPushButton:hover{background:#1976D2;}")
        b_apply.clicked.connect(self._apply)
        v.addWidget(b_apply)

        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:10px;")
        v.addWidget(self._status)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        v.addWidget(bb)
        lay.addWidget(panel)

        self._draw_adf()

    # ── ADF background ────────────────────────────────────────────────────────
    def _draw_adf(self) -> None:
        np = self._np
        adf = getattr(self._scan, "adf_cache", None) if self._scan else None
        if adf is None:
            self._img.clear()
            self._status.setText("No ADF preview for this file — draw ROIs on a blank canvas "
                                 "or load/compute the ADF first.")
            self._shape = (512, 512)
            return
        a = np.asarray(adf, dtype=float)
        self._shape = a.shape[:2]
        lo, hi = float(np.nanpercentile(a, 1)), float(np.nanpercentile(a, 99))
        self._img.setImage(a, levels=(lo, hi))
        self._vb.autoRange()

    # ── load / file switch ────────────────────────────────────────────────────
    def _load_existing(self) -> None:
        for ent in list(self._entries):
            self._vb.removeItem(ent["roi"])
        self._entries.clear()
        rois = E.scan_area_rois(self._scan) if self._scan else {}
        for rid, bounds in rois.items():
            self._make_rect(rid, bounds)
        self._rebuild_table()
        self._status.setText(f"{len(self._entries)} ROI(s) on {self._scan.name}."
                             if self._scan else "No file.")

    def _on_file_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._scans):
            self._scan = self._scans[idx]
            self._draw_adf()
            self._load_existing()

    # ── rectangle <-> table ───────────────────────────────────────────────────
    def _color_for(self, n: int) -> str:
        return self._ROI_COLORS[n % len(self._ROI_COLORS)]

    def _make_rect(self, rid: str, bounds: list) -> dict:
        pg = self._pg
        x0, x1, y0, y1 = [float(b) for b in bounds[:4]]
        color = self._color_for(len(self._entries))
        roi = pg.RectROI([x0, y0], [max(x1 - x0, 1), max(y1 - y0, 1)],
                         pen=pg.mkPen(color, width=2),
                         hoverPen=pg.mkPen(color, width=3),
                         handlePen=pg.mkPen(color, width=2))
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([1, 1], [0, 0])
        roi.setZValue(10)
        self._vb.addItem(roi)
        # a small text label at the ROI origin
        label = pg.TextItem(rid, color=color, anchor=(0, 1))
        label.setPos(x0, y0)
        label.setZValue(11)
        self._vb.addItem(label)
        # the rotation (set in the Line / ROI tool) is kept; this editor moves/resizes only
        ent = {"id": rid, "roi": roi, "label": label, "color": color,
               "angle": E.roi_angle(bounds)}
        roi.sigRegionChanged.connect(lambda _r, e=ent: self._on_rect_moved(e))
        self._entries.append(ent)
        return ent

    def _bounds_of(self, ent: dict) -> list:
        pos = ent["roi"].pos()
        size = ent["roi"].size()
        x0, y0 = float(pos.x()), float(pos.y())
        x1, y1 = x0 + float(size.x()), y0 + float(size.y())
        out = [round(x0, 1), round(x1, 1), round(y0, 1), round(y1, 1)]
        return out + [ent["angle"]] if ent.get("angle") else out

    def _on_rect_moved(self, ent: dict) -> None:
        if self._syncing:
            return
        b = self._bounds_of(ent)
        ent["label"].setPos(b[0], b[2])
        # update the matching table row
        for row in range(self._table.rowCount()):
            if self._table.item(row, 0) and self._table.item(row, 0).text() == ent["id"]:
                self._syncing = True
                for col, val in zip((1, 2, 3, 4), (b[0], b[1], b[2], b[3])):
                    it = self._table.item(row, col)
                    if it:
                        it.setText(f"{val:g}")
                self._syncing = False
                break

    def _rebuild_table(self) -> None:
        self._syncing = True
        self._table.setRowCount(len(self._entries))
        for row, ent in enumerate(self._entries):
            b = self._bounds_of(ent)
            id_item = QtWidgets.QTableWidgetItem(ent["id"])
            id_item.setFlags(id_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            id_item.setForeground(QtGui.QColor(ent["color"]))
            f = id_item.font(); f.setBold(True); id_item.setFont(f)
            self._table.setItem(row, 0, id_item)
            for col, val in zip((1, 2, 3, 4), b):
                self._table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{val:g}"))
        self._syncing = False

    def _on_cell_edited(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._syncing or item.column() == 0:
            return
        row = item.row()
        if row >= len(self._entries):
            return
        try:
            vals = [float(self._table.item(row, c).text()) for c in (1, 2, 3, 4)]
        except (ValueError, AttributeError):
            return
        x0, x1, y0, y1 = vals
        x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
        ent = self._entries[row]
        self._syncing = True
        ent["roi"].setPos([x0, y0])
        ent["roi"].setSize([max(x1 - x0, 1), max(y1 - y0, 1)])
        ent["label"].setPos(x0, y0)
        self._syncing = False

    def _on_row_selected(self) -> None:
        rows = {i.row() for i in self._table.selectedItems()}
        for row, ent in enumerate(self._entries):
            w = 3 if row in rows else 2
            ent["roi"].setPen(self._pg.mkPen(ent["color"], width=w))

    def _selected_row(self) -> int:
        rows = sorted({i.row() for i in self._table.selectedItems()})
        return rows[0] if rows else -1

    # ── add / duplicate / delete ──────────────────────────────────────────────
    def _next_roi_id(self) -> str:
        """First unused R# — counts both committed scan ROIs AND the rectangles
        already pending in this editor (so each Add gets a distinct name)."""
        taken = set(E.collect_roi_ids(self._scans)) | {e["id"] for e in self._entries}
        k = 1
        while f"R{k}" in taken:
            k += 1
        return f"R{k}"

    def _add_roi(self) -> None:
        H, W = self._shape
        rid = self._next_roi_id()
        # default rectangle: centered quarter-size box
        w, h = W / 4, H / 4
        x0, y0 = (W - w) / 2, (H - h) / 2
        self._make_rect(rid, [x0, x0 + w, y0, y0 + h])
        self._rebuild_table()
        self._status.setText(f"Added {rid}.")

    def _duplicate_roi(self) -> None:
        row = self._selected_row()
        if row < 0 or row >= len(self._entries):
            self._status.setText("Select a ROI row to duplicate.")
            return
        b = self._bounds_of(self._entries[row])
        rid = self._next_roi_id()
        off = 12.0
        self._make_rect(rid, [b[0] + off, b[1] + off, b[2] + off, b[3] + off])
        self._rebuild_table()
        self._status.setText(f"Duplicated → {rid}.")

    def _delete_roi(self) -> None:
        row = self._selected_row()
        if row < 0 or row >= len(self._entries):
            self._status.setText("Select a ROI row to delete.")
            return
        ent = self._entries.pop(row)
        self._vb.removeItem(ent["roi"])
        self._vb.removeItem(ent["label"])
        self._rebuild_table()
        self._status.setText(f"Deleted {ent['id']}.")

    # ── apply ─────────────────────────────────────────────────────────────────
    def _current_set(self) -> dict:
        return {ent["id"]: self._bounds_of(ent) for ent in self._entries}

    def _apply(self) -> None:
        if self._scan is None:
            return
        rois = self._current_set()
        log = self._host._console.log
        # write to the active (template) scan first
        self._scan.area_rois = dict(rois)
        if self._apply_all.isChecked():
            E.propagate_template_rois(self._scans, rois,
                                      use_drift=self._use_drift.isChecked(), log=log)
            scope = f"all {len(self._scans)} file(s)"
        else:
            scope = self._scan.name
        self._host._update_active_views()
        self._status.setText(f"Applied {len(rois)} ROI(s) → {scope}.")
        log(f"Area ROI editor: applied {len(rois)} ROI(s) to {scope}.")


# ─────────────────────────────────────────────────────────────────────────────
# Figure store dialog — per-step Report persistence toggles
# ─────────────────────────────────────────────────────────────────────────────

class FigureStoreDialog(QtWidgets.QDialog):
    """Checkboxes for ``DEFAULT_STORE_FIGURE`` keys (which types go to Report in report mode)."""

    def __init__(self, parent, store: dict) -> None:
        super().__init__(parent)
        _enable_minmax(self)
        self.setWindowTitle("Figure store — Report persistence")
        self.resize(420, 340)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            "<b>Which figures to keep</b> when <i>Figures</i> = report.<br>"
            "Unchecked types are discarded (or preview-only)."))
        grid = QtWidgets.QGridLayout()
        self._checks: dict[str, QtWidgets.QCheckBox] = {}
        keys = list(E.STORE_FIGURE_LABELS.keys())
        for i, key in enumerate(keys):
            cb = QtWidgets.QCheckBox(E.STORE_FIGURE_LABELS.get(key, key))
            cb.setChecked(bool(store.get(key, E.DEFAULT_STORE_FIGURE.get(key, False))))
            self._checks[key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        lay.addLayout(grid)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def values(self) -> dict:
        return {k: cb.isChecked() for k, cb in self._checks.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class Fast4DWindow(QtWidgets.QMainWindow):
    # signals delivered from worker threads to the GUI thread (queued)
    sig_progress = QtCore.Signal(object)
    sig_scan_done = QtCore.Signal(object)
    sig_finished = QtCore.Signal(object)
    sig_status = QtCore.Signal(str)
    sig_call = QtCore.Signal(object)      # run an arbitrary callable on the GUI thread

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fast4D — 4D-STEM strain & stress")
        self.resize(1280, 900)
        self.setDockNestingEnabled(True)
        self._scans: list[E.Scan] = []
        self._active = -1
        self._recent_scan_indices: list[int] = []  # LRU window for release_scans()
        self._busy = False
        self._step = STEPS[0][0]
        self._cancel_event = threading.Event()
        self._run_gen = 0               # bumped on each run + on cancel; stale finishes ignored
        self._save_root = ""
        # Export selection (Save / Save As / Export PPTX share this dialog + state)
        self._export_categories: dict = {key: True for key, _lbl in ExportSelectionDialog.CATEGORIES}
        self._template_idx = -1         # line-tool: which scan the template lines were picked on
        self._template_lines: list = []  # template line SPECS (h/v/seg), propagated to all
        self._template_roi: list = []    # template area ROI [x0,x1,y0,y1], propagated to all
        self._tools: list = []           # modeless tool windows (kept alive; pruned on open)
        self._store_figure: dict = dict(E.DEFAULT_STORE_FIGURE)

        self._build_central()
        self._build_docks()
        self._build_menu()
        self._build_icon_toolbar()
        self._build_bottom_bar()        # shares ROW 1 with workflow icons (right-aligned)
        self._build_step_toolbar()      # row 2 — step-action buttons

        self._sync_figure_policy()

        self.sig_progress.connect(self._on_progress)
        self.sig_scan_done.connect(self._on_scan_done)
        self.sig_finished.connect(self._on_finished)
        self.sig_status.connect(self._status.showMessage)
        self.sig_call.connect(lambda fn: fn())
        self._install_tqdm_sinks()
        self._btn_compute.setEnabled(False)
        self._status.showMessage("Loading py4DSTEM / GPU components in background…")
        from qt_splash import start_heavy_warmup, heavy_ready
        start_heavy_warmup(log=self._console.log)
        self._heavy_poll = QtCore.QTimer(self)
        self._heavy_poll.setInterval(200)
        self._heavy_poll.timeout.connect(self._on_heavy_ready_poll)
        self._heavy_poll.start()
        self._select_step(self._step)
        self._refresh_files()
        self._update_workflow_icons()
        self._restore_window_layout()   # user's saved panel arrangement, if any (after build)

    def _layout_settings(self) -> QtCore.QSettings:
        """Per-user panel-layout file (Files/ADF/Resources positions, sizes, floating
        state) — a real file the user can find, not the OS registry. First launch has
        no file yet, so the default arrangement built in _build_docks stands."""
        import os
        from pathlib import Path
        cfg_dir = Path(os.environ.get("APPDATA", str(Path.home()))) / "Fast4D"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return QtCore.QSettings(str(cfg_dir / "window_layout.ini"),
                                QtCore.QSettings.Format.IniFormat)

    def _restore_window_layout(self) -> None:
        settings = self._layout_settings()
        geo = settings.value("geometry")
        state = settings.value("windowState")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)

    def closeEvent(self, ev) -> None:
        """Save the user's current panel arrangement so it's restored next launch."""
        settings = self._layout_settings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        super().closeEvent(ev)

    def _on_heavy_ready_poll(self) -> None:
        from qt_splash import heavy_ready
        if not heavy_ready() or self._busy:
            return
        self._heavy_poll.stop()
        self._btn_compute.setEnabled(True)
        self._status.showMessage("Ready — load files and run Compute.")

    def _install_tqdm_sinks(self) -> None:
        """Route throttled tqdm progress to the GUI console/progress bar."""
        try:
            import qt_tqdm
        except Exception:
            return

        def progress(pct: float) -> None:
            try:
                p = int(max(0, min(100, round(float(pct)))))
            except Exception:
                return
            self.sig_call.emit(lambda p=p: self._progress.setValue(p))

        def console(line: str) -> None:
            self._console.log(str(line))

        qt_tqdm.register_sinks(progress=progress, console=console)

    # ── live accessors ─────────────────────────────────────────────────────────
    def get_scans(self) -> list[E.Scan]:
        return self._scans

    def active_scan(self) -> E.Scan | None:
        return self._scans[self._active] if 0 <= self._active < len(self._scans) else None

    # ── stdout/stderr: leave native terminal (run_gui.bat) — do NOT tee into Qt
    # ── central + docks ──────────────────────────────────────────────────────
    def _build_central(self) -> None:
        self._params = ParamTable(self.get_scans, self.active_scan)
        self.setCentralWidget(self._params)
        if self._params.report is not None:        # Report-tab Save / Save As
            self._params.report.saveRequested.connect(
                lambda save_as: self._save_results(save_as=save_as))
            self._params.report.exportPptxRequested.connect(self._export_pptx_report)
        self._params.tabStep.connect(self._select_step)   # tab → icon strip + step buttons

    def _dock(self, title, widget, area, *, name) -> QtWidgets.QDockWidget:
        d = QtWidgets.QDockWidget(title, self)
        d.setObjectName(name)
        d.setWidget(widget)
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                      | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        widget.setMinimumSize(160, 80)             # no infinite collapse
        self.addDockWidget(area, d)
        return d

    def _build_docks(self) -> None:
        L = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        R = QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        B = QtCore.Qt.DockWidgetArea.BottomDockWidgetArea

        # Give the bottom CORNERS to the side areas so the bottom (Console) dock spans
        # ONLY the central column (under the figures / param table) — NOT under the
        # Files / Resources column (left) nor the ADF column (right). The left and
        # right docks then run the full window height, leaving more room for the ADF.
        self.setCorner(QtCore.Qt.Corner.BottomLeftCorner, L)
        self.setCorner(QtCore.Qt.Corner.BottomRightCorner, R)
        self.setCorner(QtCore.Qt.Corner.TopLeftCorner, L)
        self.setCorner(QtCore.Qt.Corner.TopRightCorner, R)

        # Files dock
        files = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(files)
        fl.setContentsMargins(4, 4, 4, 4)
        btn_host = QtWidgets.QWidget()
        btn_host.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                               QtWidgets.QSizePolicy.Policy.Maximum)
        btn_flow = FlowLayout(btn_host, margin=0, hspacing=6, vspacing=6)
        # Primary buttons — the four actions every user needs every session.
        for txt, fn, tip in (
                ("Load…", self._open_loader,
                 "Unified loader: samples + braggpeaks + ADF + JSON + vacuum/probe"),
                ("Save…", self._save_session_dialog,
                 "Save results (one folder per file: data/ + figures/) + session JSON"),
                ("Refresh", self._refresh_data,
                 "Re-load saved strain (with/without ROI) for workspace files"),
                ("Remove", self._remove_active, "Remove the selected file")):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            b.setToolTip(tip)
            btn_flow.addWidget(b)
        # Secondary actions — power-user tools in a single "⋮" menu to keep the
        # Files dock uncluttered without removing any functionality.
        b_more = QtWidgets.QPushButton("⋮ More")
        b_more.setToolTip("Advanced file actions: save params, embed calibrations, load calib from h5…")
        def _show_more_menu():
            menu = QtWidgets.QMenu(b_more)
            for txt, fn, tip in (
                    ("Save params…", self._save_params_template,
                     "Save ONLY the parameter table → Parametros_cal.json (re-importable template)"),
                    ("Embed calibrations → h5", self._embed_metadata,
                     "Embed the FULL analysis (params/calibrations/ROI/lines/drift/ranges) as "
                     "metadata INSIDE each file's .h5 — self-describing, no JSON needed to reopen."),
                    ("Load calib from h5…", self._load_metadata_from_h5,
                     "Browse for any .h5 and restore calibrations from its fast4d_metadata "
                     "into the selected scan (also sets virtual h5 path when empty).")):
                act = menu.addAction(txt)
                act.setToolTip(tip)
                act.triggered.connect(fn)
            menu.exec(b_more.mapToGlobal(b_more.rect().bottomLeft()))
        b_more.clicked.connect(_show_more_menu)
        btn_flow.addWidget(b_more)
        fl.addWidget(btn_host)
        # colormap for the saved virtual-image PNGs (ADF/BF/DP) — default grayscale
        cmaprow = QtWidgets.QHBoxLayout()
        cmaprow.addWidget(QtWidgets.QLabel("Saved image cmap:"))
        self._cmap = QtWidgets.QComboBox(); self._cmap.addItems(list(E.VIMG_CMAPS))
        self._cmap.setCurrentText("gray")
        self._cmap.setToolTip("Colormap used when saving the ADF/BF/DP virtual images as PNG "
                              "(the raw .npy is always saved too).")
        cmaprow.addWidget(self._cmap, 1)
        fl.addLayout(cmaprow)
        hint = QtWidgets.QLabel("Ctrl+click several files to queue them · "
                                "Expand a file to explore its h5 root · double-click "
                                "an image node to view it (no heavy load).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:9px;")
        fl.addWidget(hint)
        self._files = FilesTree(explore_fn=E.explore_h5)
        self._files.scanSelected.connect(self._on_file_selected)
        self._files.nodeActivated.connect(self._on_node_activated)
        # Clicking a file's column in the middle parameter table must select that
        # same file in the Files panel too — Play Calibration/Analysis always acts
        # on the Files-panel selection (self._active), so without this the two
        # views could point at different files and an action would silently run
        # on the wrong one.
        self._params.fileSelected.connect(self._files.select_scan)
        fl.addWidget(self._files, 1)
        self._files_dock = self._dock("Files", files, L, name="files")

        # ADF dock + ADF gallery (tabbed with it) — user-requested layout: everything
        # stacked on the LEFT, top to bottom: Files / ADF viewer+gallery / Resources.
        self._adf = AdfView()
        self._adf.popoutRequested.connect(self._popout_adf)
        self._adf_dock = self._dock("ADF viewer", self._adf, L, name="adf")
        self._gallery = AdfGallery()
        self._gallery.selected.connect(self._on_gallery_selected)
        self._gallery.loadAllRequested.connect(self._load_all_adfs)
        self._gallery_dock = self._dock("ADF Gallery", self._gallery, L, name="gallery")
        self.tabifyDockWidget(self._adf_dock, self._gallery_dock)
        self._adf_dock.raise_()
        # (Report now lives as the LAST tab of the central calibration table, after Stress.)

        # Status dock (resources + calstate)
        status = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(status)
        sl.setContentsMargins(4, 4, 4, 4)
        self._res = ResourceMonitor()
        self._cal = CalStateStrip()
        sl.addWidget(self._res)
        b_free = QtWidgets.QPushButton("Clear figures")
        b_free.setToolTip(
            "Drop preview figure slots and close orphan matplotlib windows. "
            "Report figures (origin, strain, …) are kept.")
        b_free.clicked.connect(self._clear_figures)
        sl.addWidget(b_free)
        b_free_ram = QtWidgets.QPushButton("Free RAM")
        b_free_ram.setToolTip("Release the heavy datacube (.mib) + intermediate buffers and the "
                              "CUDA memory pool after a compute. The ADF preview + calibration "
                              "survive; the datacube re-loads on demand.")
        b_free_ram.clicked.connect(self._free_ram)
        sl.addWidget(b_free_ram)
        sl.addWidget(self._cal)
        sl.addStretch(1)
        self._status_dock = self._dock("Resources / Calibration state", status, L, name="status")

        # Explicit vertical stacking (top → bottom): Files, then ADF (+ Gallery tab), then
        # Resources/Calibration — don't rely on Qt's implicit same-area insertion order.
        self.splitDockWidget(self._files_dock, self._adf_dock, QtCore.Qt.Orientation.Vertical)
        self.splitDockWidget(self._adf_dock, self._status_dock, QtCore.Qt.Orientation.Vertical)

        # Console dock (bottom)
        self._console = ConsoleWidget()
        self._console_dock = self._dock("Console — py4DSTEM log", self._console, B, name="console")

    # ── menu (View → reopen panels) ─────────────────────────────────────────────
    def _build_menu(self) -> None:
        """View → Tools → Settings → Help menu bar (always visible)."""
        mb = self.menuBar()
        mb.setVisible(True)
        mb.setNativeMenuBar(False)   # keep menus in-window on Windows (not hidden in title bar)
        self._docks = [self._files_dock, self._status_dock, self._adf_dock,
                       self._gallery_dock, self._console_dock]
        view = mb.addMenu("&View")
        for dock in self._docks:
            view.addAction(dock.toggleViewAction())   # checkable show/hide → reopens it
        view.addSeparator()
        act = QtGui.QAction("Show all panels", self)
        act.triggered.connect(self._show_all_docks)
        view.addAction(act)

        tools = mb.addMenu("&Tools")
        act_measure = QtGui.QAction("Line / ROI profiles…", self)
        act_measure.setToolTip(
            "One tool for line profiles and (rotatable) ROI statistics on any map.")
        act_measure.triggered.connect(self._open_live_line)
        tools.addAction(act_measure)
        tools.addSeparator()
        act_setup = QtGui.QAction("Set up Lines & ROI…", self)
        act_setup.setToolTip(
            "Configure line ROIs, area ROIs, drift, and propagation.")
        act_setup.triggered.connect(self._open_line_setup)
        tools.addAction(act_setup)
        act_drift = QtGui.QAction("Inter-scan drift (ADF / BF)…", self)
        act_drift.setToolTip(
            "Rigid drift between repeated scans of the same region, from their ADF/BF; "
            "writes the per-file drift CSV used by the line / ROI tools.")
        act_drift.triggered.connect(lambda: self._show_tool(DriftEstimateDialog(self)))
        tools.addAction(act_drift)
        act_repro = QtGui.QAction("Reproducibility experiment…", self)
        act_repro.setToolTip(
            "Same region scanned several times: line / ROI / map repeatability, "
            "each without and with the inter-scan drift correction.")
        act_repro.triggered.connect(lambda: self._show_tool(ReproducibilityDialog(self)))
        tools.addAction(act_repro)
        tools.addSeparator()
        act_an_file = QtGui.QAction("Analyse (file)", self)
        act_an_file.setToolTip(
            "Run full analysis (stress + lines) on the active file.")
        act_an_file.triggered.connect(self._analyze_active)
        tools.addAction(act_an_file)
        act_an_all = QtGui.QAction("Analysis (all)", self)
        act_an_all.setToolTip(
            "Run full analysis (stress + lines) on all loaded files.")
        act_an_all.triggered.connect(lambda: self._on_analysis())
        tools.addAction(act_an_all)

        settings_m = mb.addMenu("&Settings")
        act_mem = QtGui.QAction("Resident data (RAM)…", self)
        act_mem.setToolTip(
            "Configure how many scans stay fully resident in RAM "
            "when switching scans.")
        act_mem.triggered.connect(self._configure_resident_data_policy)
        settings_m.addAction(act_mem)
        self._cb_save = QtGui.QAction("Auto-save on Compute", self)
        self._cb_save.setCheckable(True)
        self._cb_save.setChecked(True)
        self._cb_save.setToolTip(
            "ON (default): every Compute (File or All) saves the workspace + "
            "figures automatically, same as before.\n"
            "OFF: Compute only runs the calculation — nothing is written to disk "
            "until you Save manually (Report panel Save / Save As). Skips the "
            "figure-overwrite I/O on every calculation, which is slow for "
            "large/frequent recomputes.")
        settings_m.addAction(self._cb_save)

        help_m = mb.addMenu("&Help")
        act_qs = QtGui.QAction("Quick Start Guide…", self)
        act_qs.setToolTip("5-step introduction: data types, workflow, fastest path to strain maps.")
        act_qs.triggered.connect(self._show_quick_start)
        help_m.addAction(act_qs)
        help_m.addSeparator()
        act_guide = QtGui.QAction("Calibration guide…", self)
        act_guide.setToolTip("Workflow help: overlays, RAM, Compute fit/apply, step jumps.")
        act_guide.triggered.connect(self._show_calib_guide)
        help_m.addAction(act_guide)

    def _sync_figure_policy(self) -> None:
        kw = dict(
            mode=self._fig_mode.currentText(),
            max_in_ram=int(self._fig_max.value()),
            close_orphans=True,
            spill_to_disk=self._cb_spill.isChecked(),
            store=dict(self._store_figure),
            spill_dpi=int(self._spill_dpi.value()),
            save_dpi=int(self._save_dpi.value()),
        )
        E.set_figure_policy(**kw)

    def _open_figure_store(self) -> None:
        dlg = FigureStoreDialog(self, self._store_figure)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._store_figure = dlg.values()
            self._sync_figure_policy()
            self._console.log("Figure store updated — affects next register / Compute.")

    def _configure_resident_data_policy(self) -> None:
        """Settings → Memory → Max scans kept in RAM."""
        current = E.get_data_policy().max_scans_in_ram
        n, ok = QtWidgets.QInputDialog.getInt(
            self, "Memory settings",
            "Max scans kept fully resident in RAM\n"
            "(others release their datacube/BVM/probe on scan-switch;\n"
            "figures, ADF previews, and braggpeaks are unaffected):",
            current, 1, 20, 1)
        if ok:
            E.set_data_policy(max_scans_in_ram=n)
            self._console.log(f"Resident-data policy: max_scans_in_ram={n}")

    def _clear_figures(self) -> None:
        if self._busy:
            return
        self._sync_figure_policy()
        res = E.tidy_figure_memory(self._scans, log=self._console.log)
        self._params.report_refresh()
        self._status.showMessage(res.get("status", "Figures cleared."))

    def _maybe_tidy_figures(self) -> None:
        mode = self._fig_mode.currentText()
        if mode in ("off", "preview"):
            E.tidy_figure_memory(self._scans, log=None)
        elif E.pyplot_figure_count() > 18:
            self._console.log(
                f"[warn] {E.figure_memory_status(self._scans)} — "
                "try Figures=off/preview or Clear figures.")

    def _show_all_docks(self) -> None:
        for dock in getattr(self, "_docks", []):
            dock.show()

    def _show_quick_start(self) -> None:
        from qt_quickstart import QuickStartDialog
        dlg = QuickStartDialog(self)
        dlg.exec()

    def _show_calib_guide(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Calibration guide — fast4d")
        dlg.resize(560, 520)
        lay = QtWidgets.QVBoxLayout(dlg)
        browser = QtWidgets.QTextBrowser()
        browser.setHtml(_CALIB_GUIDE_HTML)
        browser.setOpenExternalLinks(True)
        lay.addWidget(browser, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def _ensure_console_visible(self) -> None:
        """Bring the Console dock back if the user closed it."""
        dock = getattr(self, "_console_dock", None)
        if dock is None:
            return
        if not dock.isVisible():
            dock.show()
        dock.raise_()

    def _open_calib_tool(self, step_key: str) -> None:
        meta = _CALIB_META.get(step_key)
        if not meta:
            return
        _meth = meta[1]
        if self._need_active() is None:
            return
        getattr(self, _meth)()

    def _run_calibration_prep(self, sc, prev_step: str | None, *, title: str,
                              on_done) -> None:
        """Apply upstream calibrations with a live log window, then call ``on_done``."""
        from qt_widgets import CalibLoadingDialog
        dlg = CalibLoadingDialog(self, title=f"Loading calibration — {sc.name}")
        self._ensure_console_visible()
        dlg.show()
        dlg.raise_()

        def log(msg: str) -> None:
            dlg.append_log(msg)
            self._console.log(msg)

        def work():
            log(f"[{sc.name}] Reading braggpeaks…")
            E.ensure_braggpeaks_for_calibration(sc, log=log)
            if prev_step:
                upstream = E.CHECKPOINT_ORDER[:E.CHECKPOINT_ORDER.index(prev_step)]
                if upstream:
                    log(f"[{sc.name}] Applying from parameter table: "
                        + " → ".join(upstream) + " …")
                else:
                    log(f"[{sc.name}] No upstream calibration steps before {title}.")
                steps = E.apply_calibrations_through(sc, prev_step, log=log)
                log(f"[{sc.name}] Upstream done: {', '.join(steps) or '(none)'}")
            else:
                steps = []
            return steps

        def finished(result):
            dlg.close()
            if isinstance(result, Exception):
                QtWidgets.QMessageBox.warning(
                    self, "Loading calibration",
                    f"Could not prepare '{sc.name}':\n{result}")
                return
            self._params.reload()
            self._update_active_views()
            if on_done:
                on_done()

        self._run_async(work, label=f"Loading calib ({sc.name})", on_done=finished)

    def _setting_calibration(self, step_key: str) -> None:
        """Setting X calibration — check status, prep upstream if needed, open tool."""
        meta = _CALIB_META.get(step_key)
        if not meta:
            return
        sc = self._need_active()
        if sc is None or self._busy:
            return
        prev_step, _meth = meta
        title = _CALIB_TITLES.get(step_key, step_key)
        setting_label = LBL_SETTING_CALIB.format(name=title)

        try:
            E.ensure_braggpeaks_for_calibration(sc, log=self._console.log)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, setting_label,
                f"Cannot open calibration for '{sc.name}':\n{exc}")
            return

        self._params.apply()

        if step_key == "roi":
            self._open_calib_tool(step_key)
            return

        status = E.calibration_step_status(sc, step_key)
        if step_key == "ellipse" and not sc.params.ellipse_enabled:
            QtWidgets.QMessageBox.information(
                self, setting_label,
                f"Ellipse is disabled for '{sc.name}'.\n\n"
                "Check <b>Enabled</b> in the Ellipse parameter table, then Apply.")
            return

        if status == "applied":
            ans = QtWidgets.QMessageBox.question(
                self, setting_label,
                f"<b>{title}</b> is already calibrated for '{sc.name}'.\n\n"
                "Open the setting tool anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes)
            if ans == QtWidgets.QMessageBox.StandardButton.Yes:
                self._open_calib_tool(step_key)
            return

        # Not calibrated (pending / staged)
        if prev_step is None:
            msg = (f"<b>{title}</b> is not calibrated yet for '{sc.name}'.\n\n"
                   "Open the setting tool?")
        else:
            msg = (f"<b>{title}</b> is not calibrated yet for '{sc.name}'.\n\n"
                   "Apply previous calibration steps from the parameter table, "
                   "then open the tool?")
        ans = QtWidgets.QMessageBox.question(
            self, setting_label, msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes)
        if ans != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        if prev_step is None:
            self._open_calib_tool(step_key)
        else:
            self._run_calibration_prep(
                sc, prev_step, title=title,
                on_done=lambda: self._open_calib_tool(step_key))

    # ── toolbars ──────────────────────────────────────────────────────────────
    def _icon(self, stem: str) -> QtGui.QIcon:
        p = icon_path(stem)
        return QtGui.QIcon(str(p)) if p.is_file() else QtGui.QIcon()

    def _build_icon_toolbar(self) -> None:
        tb = QtWidgets.QToolBar("Workflow")
        tb.setObjectName("workflow_toolbar")
        tb.setIconSize(QtCore.QSize(64, 64))      # big workflow icons (user request)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        tb.setStyleSheet(
            "QToolBar#workflow_toolbar{background:#D6ECFF;}"
            "QToolBar#workflow_toolbar QToolButton{background:#E1F5FE;border:1px solid #B3E5FC;"
            "border-radius:10px;padding:6px;margin:3px;color:#0D47A1;font-size:10px;}"
            "QToolBar#workflow_toolbar QToolButton:hover{background:#B3E5FC;}"
            "QToolBar#workflow_toolbar QToolButton:checked{background:#4FC3F7;border:2px solid #0277BD;}"
            "QToolBar#workflow_toolbar QToolButton#boxed{background:#FFF3E0;border:2px solid #FB8C00;}"
            "QToolBar#workflow_toolbar QToolButton#boxed:checked{background:#FFB74D;border:2px solid #E65100;}")
        tb.setMovable(True)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)
        self._workflow_tb = tb
        # "Load Data" — the unified loader — sits top-left, before the steps.
        load_act = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton),
            "Load Data", self)
        load_act.setToolTip("Open the unified loader (samples / braggpeaks / ADF / JSON / vacuum-probe)")
        load_act.triggered.connect(self._open_loader)
        tb.addAction(load_act)
        tb.addSeparator()
        self._step_actions_group: dict[str, QtGui.QAction] = {}
        for key, label, stem in STEPS:
            act = QtGui.QAction(self._icon(stem), label, self)
            act.setCheckable(True)
            act.setToolTip(_STEP_TOOLTIPS.get(key, label))
            act.triggered.connect(lambda _c=False, k=key: self._select_step(k))
            tb.addAction(act)
            self._step_actions_group[key] = act
        tb.style().unpolish(tb); tb.style().polish(tb)

    def _build_step_toolbar(self) -> None:
        self._step_tb = QtWidgets.QToolBar("Step actions")
        self._step_tb.setObjectName("step_toolbar")
        self._step_tb.setMovable(True)
        self.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self._step_tb)

    def _update_workflow_icons(self) -> None:
        """Dim Probe / 6 Points / Detection when the active file is Path A."""
        sc = self.active_scan()
        path_a = sc is not None and E.analysis_path(sc) == "A"
        tb = getattr(self, "_workflow_tb", None)
        for key, act in self._step_actions_group.items():
            dim = path_a and key in _PATH_B_STEPS
            base = _STEP_TOOLTIPS.get(key, act.text())
            if dim:
                act.setToolTip(
                    f"{base}\n\nPath A ({sc.name}): braggpeaks.h5 loaded — "
                    "optional; skip to ROI → Origin.")
            else:
                act.setToolTip(base)
            if tb is None:
                continue
            w = tb.widgetForAction(act)
            if w is None:
                continue
            if dim:
                w.setStyleSheet("opacity:0.40; color:#78909C; font-size:9px;")
            else:
                w.setStyleSheet("")

    def _add_path_banner(self, tb, *, path: str) -> None:
        """Path-A hint in the step toolbar (detection steps only)."""
        if path != "A":
            return
        lbl = QtWidgets.QLabel(f"  {PATH_A_DET_MSG}  ")
        lbl.setStyleSheet("color:#1B5E20; font-size:10px; font-weight:600;")
        lbl.setToolTip(PATH_A_DET_TIP)
        tb.addWidget(lbl)
        tb.addSeparator()

    def _path_b_only(self, sc, action: str) -> bool:
        """Return True if Path-B action may proceed; False after informing on Path A."""
        if sc is None or E.needs_detection_workflow(sc):
            return True
        QtWidgets.QMessageBox.information(
            self, "Path A — skip detection",
            f"'{sc.name}' already has braggpeaks.h5.\n\n"
            f"{action} is not needed for calibration — use ROI → Origin → … directly.\n\n"
            "To re-detect from raw, remove or replace the braggpeaks file first.")
        return False

    def _auto_load_braggpeaks(self, sc) -> None:
        """Path A braggpeaks load. NOT wired to selection any more — selecting a
        file/ADF must stay cheap (preview only). Kept for explicit callers; the
        braggpeaks.h5 read goes through py4DSTEM and is too slow to run on click."""
        if sc is None or E.analysis_path(sc) != "A":
            return
        if getattr(getattr(sc, "state", None), "braggpeaks", None) is not None:
            return
        try:
            E.ensure_braggpeaks_for_calibration(sc, log=self._console.log)
        except Exception as exc:
            self._console.log(f"[{sc.name}] braggpeaks auto-load: {exc}")

    def _populate_step_actions(self, key: str) -> None:
        tb = self._step_tb
        tb.clear()
        def add(text, fn, *, tip: str = "", style: str = ""):
            b = QtWidgets.QPushButton(text)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            ss = ("QPushButton{padding:5px 12px; margin:2px; border:1px solid #1565C0;"
                  "border-radius:5px; background:#E3F2FD; color:#0D47A1; font-weight:600;}"
                  "QPushButton:hover{background:#BBDEFB;}"
                  "QPushButton:pressed{background:#90CAF9;}")
            if style:
                ss = style
            b.setStyleSheet(ss)
            if tip:
                b.setToolTip(tip)
            # QPushButton.clicked passes a bool — must not forward it into step_key args.
            b.clicked.connect(lambda _checked=False, f=fn: f())
            tb.addWidget(b)
            return b
        def add_setting_bar(step_key: str, *, extra=None) -> None:
            """Setting … calibration · Apply · Reset (same active file)."""
            name = _CALIB_TITLES.get(step_key, step_key)
            add(LBL_SETTING_CALIB.format(name=name),
                lambda sk=step_key: self._setting_calibration(sk),
                tip=f"Check calibration status, apply upstream steps if needed, "
                    f"then open the {name} tool.",
                style=_SETTING_BTN_STYLE)
            if extra:
                for text, fn, tip in extra:
                    add(text, fn, tip=tip or "")
            add(LBL_APPLY_CALIB, lambda sk=step_key: self._apply_calib_step(sk), tip=TIP_APPLY)
            add(LBL_RESET, lambda sk=step_key: self._reset_calib_step(sk), tip=TIP_RESET)
            tb.addSeparator()
        sc = self.active_scan()
        path = E.analysis_path(sc)
        if key == "probe":
            self._add_path_banner(tb, path=path)
            add("Virtual ADF/BF", self._open_virtualization,
                tip="Build ADF/BF/DP mean/max virtual images from the raw datacube "
                    "(place detectors → compute → save .h5). Available regardless "
                    "of calibration path.")
            if path == "B":
                add("Choose vacuum file…", self._pick_vacuum)
                add("Pick vacuum region…", self._pick_vacuum_region)
                add("Load data (heavy)", lambda: self._async_step("load_datacube"))
                add("Compute probe", self._compute_probe)
        elif key == "select6":
            self._add_path_banner(tb, path=path)
            if path == "B":
                add("Pick 6 points…", self._pick_six)
        elif key == "detection":
            self._add_path_banner(tb, path=path)
            if path == "B":
                add("Load data + probe", self._load_data_and_probe)
                add("Tune detection (live)…", self._open_detect_tuner)
                add("Preview at 6 pts", lambda: self._async_step("detect_preview"))
            if any(E.needs_detection_workflow(s) for s in self._scans):
                add("Compute braggpeaks…", self._compute_braggpeaks,
                    tip="Ctrl+click several files in Files to queue them. Uses the "
                        "detection parameters of the current (tuned) file. Load data "
                        "+ probe runs automatically per file, then RAM is freed.")
        elif key == "roi":
            add_setting_bar("roi")
            add("Pick on ADF…", self._pick_roi, tip="Quick ROI pick on the active ADF.")
        elif key == "origin":
            add_setting_bar("origin")
        elif key == "ellipse":
            add_setting_bar("ellipse")
        elif key == "qpixel":
            add_setting_bar("qpixel", extra=[
                ("Crystal…", self._open_crystal_editor,
                 "Edit the Q-pixel calibration crystal (element, structure, a)."),
            ])
        elif key == "basis":
            add_setting_bar("basis")
        elif key == "strain":
            add("Apply", self._apply_strain_params, tip="Commit strain parameters from the table.")
            add("Compute (file)", self._compute_active)
            add("Compute (all)", lambda: self._on_compute())
            # Analyse / Live tools → menu Tools
        else:
            lbl = QtWidgets.QLabel("  Edit parameters in the table, then Compute. ")
            lbl.setStyleSheet("color:#888;")
            tb.addWidget(lbl)

    def _select_step(self, key: str) -> None:
        self._step = key
        for k, act in self._step_actions_group.items():
            act.setChecked(k == key)
        self._params.show_step(key)
        self._populate_step_actions(key)

    def _build_bottom_bar(self) -> None:
        # Run strip — options on the first row, actions on the second row.
        tb = QtWidgets.QToolBar("Run")
        tb.setObjectName("run_toolbar")
        tb.setMovable(True)
        tb.setStyleSheet("QToolBar#run_toolbar { spacing: 4px; }")
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)
        self._run_tb = tb
        lead = QtWidgets.QWidget()
        lead.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Preferred)
        tb.addWidget(lead)

        _lbl = "font-size:9px; color:#444;"
        _cb = "font-size:9px;"

        panel = QtWidgets.QWidget()
        panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                            QtWidgets.QSizePolicy.Policy.Fixed)
        root = QtWidgets.QHBoxLayout(panel)
        root.setContentsMargins(0, 1, 0, 1)
        root.setSpacing(6)

        def group_widget() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
            w = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(3)
            return w, v

        def hrow() -> QtWidgets.QHBoxLayout:
            h = QtWidgets.QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            return h

        def add_sep() -> None:
            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            sep.setStyleSheet("QFrame{border-left:2px dashed #1B1B1B; margin:0 3px;}")
            root.addWidget(sep)

        # Group 1: figure/save/calibration options → reset calibration action.
        g_file, g_file_v = group_widget()
        file_top = hrow()
        b_guide = QtWidgets.QToolButton()
        b_guide.setText("?")
        b_guide.setFixedWidth(22)
        b_guide.setToolTip("Open the calibration workflow guide.")
        b_guide.clicked.connect(self._show_calib_guide)
        file_top.addWidget(b_guide)
        file_top.addWidget(QtWidgets.QLabel("Figures:"))
        self._fig_mode = QtWidgets.QComboBox()
        self._fig_mode.addItems(["report", "preview", "off"])
        self._fig_mode.setFixedWidth(82)
        self._fig_mode.setToolTip(
            "report = keep figures for the Report (Compute).\n"
            "preview = show in tools only — do not register (saves RAM).\n"
            "off = discard figures immediately while testing.")
        self._fig_mode.currentIndexChanged.connect(lambda _i: self._sync_figure_policy())
        file_top.addWidget(self._fig_mode)
        self._fig_max = QtWidgets.QSpinBox()
        self._fig_max.setRange(4, 40)
        self._fig_max.setValue(12)
        self._fig_max.setFixedWidth(64)
        self._fig_max.setToolTip("Maximum registered figures kept per scan.")
        self._fig_max.valueChanged.connect(lambda _v: self._sync_figure_policy())
        file_top.addWidget(self._fig_max)
        g_file_v.addLayout(file_top)

        file_mid = hrow()
        file_mid.addWidget(QtWidgets.QLabel("Calib:"))
        self._calib = QtWidgets.QComboBox()
        self._calib.addItems(["fit", "apply"])
        self._calib.setFixedWidth(68)
        self._calib.setToolTip(
            "fit = recalibrate origin → ellipse → Q-pixel → basis (full refit).\n"
            "apply = re-apply known table values (fast, no refit).\n\n"
            "New file → fit. Strain-only rerun → apply.\n"
            "Help → Calibration guide for the full workflow.")
        file_mid.addWidget(self._calib)
        g_file_v.addLayout(file_mid)

        file_bottom = hrow()
        self._btn_reset_cal = QtWidgets.QPushButton("↺ Reset calibration")
        self._btn_reset_cal.setToolTip(
            "Reload braggpeaks for the selected file and clear origin / ellipse / "
            "Q-pixel / basis calibrations (back to uncalibrated).")
        self._btn_reset_cal.clicked.connect(self._reset_bragg_cal)
        file_bottom.addWidget(self._btn_reset_cal)
        g_file_v.addLayout(file_bottom)
        root.addWidget(g_file)
        add_sep()

        # Group 2: ROI options → Compute action.
        g_compute, g_compute_v = group_widget()
        compute_top = hrow()
        lbl_s = QtWidgets.QLabel("Strain:")
        lbl_s.setStyleSheet(_lbl)
        self._cb_strain_roi = QtWidgets.QCheckBox("ROI")
        self._cb_strain_roi.setChecked(True)
        self._cb_strain_roi.setStyleSheet(_cb)
        self._cb_strain_roi.setToolTip(
            "Also compute strain with the calibration ROI as g₁,g₂ reference "
            "(full-scan strain always runs).")
        lbl_t = QtWidgets.QLabel("Stress:")
        lbl_t.setStyleSheet(_lbl)
        self._cb_stress_roi = QtWidgets.QCheckBox("ROI")
        self._cb_stress_roi.setChecked(True)
        self._cb_stress_roi.setStyleSheet(_cb)
        self._cb_stress_roi.setToolTip(
            "Also compute Hooke stress from the with-ROI strain map "
            "(full-field stress always runs when strain exists).")
        for w in (lbl_s, self._cb_strain_roi, lbl_t, self._cb_stress_roi):
            compute_top.addWidget(w)
        g_compute_v.addLayout(compute_top)
        compute_bottom = hrow()
        compute_bottom.addStretch(1)
        self._btn_compute = QtWidgets.QPushButton("▶ Compute")
        self._btn_compute.clicked.connect(lambda: self._on_compute())
        compute_bottom.addWidget(self._btn_compute)
        compute_bottom.addStretch(1)
        g_compute_v.addLayout(compute_bottom)
        root.addWidget(g_compute)
        add_sep()

        # Group 3: figure storage / DPI options (Analysis actions → menu Tools).
        g_store, g_store_v = group_widget()
        store_top = hrow()
        self._cb_spill = QtWidgets.QCheckBox("Spill")
        self._cb_spill.setChecked(True)
        self._cb_spill.setToolTip(
            "When the RAM cap evicts a figure, write a PNG sidecar first "
            "(still visible in Report with dashed border).")
        self._cb_spill.toggled.connect(lambda _on: self._sync_figure_policy())
        store_top.addWidget(self._cb_spill)
        store_top.addWidget(QtWidgets.QLabel("View DPI:"))
        self._spill_dpi = QtWidgets.QSpinBox()
        self._spill_dpi.setRange(48, 180)
        self._spill_dpi.setValue(72)
        self._spill_dpi.setFixedWidth(78)
        self._spill_dpi.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self._spill_dpi.setToolTip(
            "PNG resolution for spilled / temp sidecars (GUI viewing only).")
        self._spill_dpi.valueChanged.connect(lambda _v: self._sync_figure_policy())
        store_top.addWidget(self._spill_dpi)
        store_top.addWidget(QtWidgets.QLabel("Save DPI:"))
        self._save_dpi = QtWidgets.QSpinBox()
        self._save_dpi.setRange(72, 400)
        self._save_dpi.setValue(300)
        self._save_dpi.setFixedWidth(82)
        self._save_dpi.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self._save_dpi.setToolTip(
            "PNG resolution for figures/ on Compute and manual Save.")
        self._save_dpi.valueChanged.connect(lambda _v: self._sync_figure_policy())
        store_top.addWidget(self._save_dpi)
        g_store_v.addLayout(store_top)
        # Which files the Report's Session analysis uses is chosen there
        # (Session analysis > Files + Refresh), not by a global checkbox.
        store_bottom = hrow()
        store_bottom.addStretch(1)
        b_store = QtWidgets.QPushButton("Store…")
        b_store.setToolTip("Choose which figure types are saved to the Report (report mode).")
        b_store.clicked.connect(self._open_figure_store)
        store_bottom.addWidget(b_store)
        store_bottom.addStretch(1)
        g_store_v.addLayout(store_bottom)
        root.addWidget(g_store)

        for b in (self._btn_reset_cal, self._btn_compute, b_store):
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                            QtWidgets.QSizePolicy.Policy.Fixed)

        tb.addWidget(panel)
        self._progress = self._params.progress_bar
        self._btn_cancel = QtWidgets.QPushButton("■ Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setStyleSheet("QPushButton:enabled{color:#C62828; font-weight:600;}")
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_cancel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                       QtWidgets.QSizePolicy.Policy.Fixed)
        self._params.add_progress_widget(self._btn_cancel)

        self._status = self.statusBar()
        self._status.showMessage("Ready.")

    # ── file management ─────────────────────────────────────────────────────────
    def _add_raw(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add raw 4D data", "",
            "4D-STEM (*.mib *.dm4 *.h5 *.hdf5 *.emd);;All (*)")
        for p in paths:
            bp = E.find_sidecar_braggpeaks(p)        # <stem>braggpeaks.h5 in same folder
            sc = E.Scan(name=Path(p).stem, raw_path=p, braggpeaks_path=(bp or ""),
                        h5_path=E.find_sidecar_h5(p))   # <stem>.h5 light preview sibling
            self._scans.append(sc)
            if bp:
                self._console.log(f"[{sc.name}] braggpeaks auto-detected → {Path(bp).name} "
                                  f"(Path A: calibrate only, no recompute)")
            else:
                self._console.log(f"[{sc.name}] no braggpeaks found → Path B (build from raw)")
        if paths:
            self._scans_changed()

    def _add_bragg(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add braggpeaks.h5 (Path A)", "", "HDF5 (*.h5 *.hdf5);;All (*)")
        for p in paths:
            self._scans.append(E.Scan(name=Path(p).stem, braggpeaks_path=p, h5_path=p))
        if paths:
            self._scans_changed()

    def _load_template(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load template (Parametros_cal.json)", "", "JSON (*.json);;All (*)")
        if not p:
            return
        try:
            self._scans = E.scans_from_template(p)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Template", f"Could not load:\n{exc}")
            return
        self._console.log(f"Template: {len(self._scans)} scan(s) from {Path(p).name}")
        self._active = 0 if self._scans else -1
        self._recent_scan_indices = []
        self._scans_changed()

    def _open_loader(self) -> None:
        """The unified, smart Load dialog (replaces the many separate buttons)."""
        from qt_loader import LoaderDialog
        dlg = LoaderDialog(self, self._scans)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._scans = dlg.scans
            self._active = 0 if self._scans else -1
            self._recent_scan_indices = []
            self._console.log(f"Loaded {len(self._scans)} scan(s) via unified loader.")
            self._scans_changed()
            self._load_all_adfs()          # load ADFs immediately (from braggpeaks h5)

    def _braggpeaks_reloadable(self, scan) -> bool:
        """True when a scan's detected peaks are persisted to disk, so dropping them
        from RAM is free — ensure_braggpeaks_for_calibration re-loads from the .h5."""
        bp = getattr(scan, "braggpeaks_path", None)
        try:
            return bool(bp) and Path(str(bp)).exists()
        except Exception:
            return False

    def _free_ram(self) -> None:
        """Release the heavy datacube + buffers + CUDA pool after a compute (RAM stays
        full otherwise, e.g. a 33 GB .mib). Runs in a worker so the UI stays live.

        When any loaded scan's peaks are persisted to a braggpeaks .h5, offer to also
        drop the peaks from RAM — they reload from disk on demand (Path A)."""
        if self._busy:
            return
        drop_bp = False
        if any(self._braggpeaks_reloadable(s) for s in self._scans if s is not None):
            ans = QtWidgets.QMessageBox.question(
                self, "Free RAM",
                "Also drop detected Bragg peaks from RAM?\n\n"
                "They are saved to disk and will reload automatically from the "
                "braggpeaks .h5 when calibration/strain next needs them.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            drop_bp = (ans == QtWidgets.QMessageBox.StandardButton.Yes)
        self._run_async(
            lambda: E.free_memory(self._scans, drop_braggpeaks=drop_bp, log=self._console.log),
            label="Free RAM",
            on_done=lambda r: self._status.showMessage(
                f"Freed {r.get('buffers', 0)} buffer(s)"
                + (f" + {r['gpu_freed_bytes']/1e6:.0f} MB GPU" if r.get('gpu_freed_bytes') else "")
                if isinstance(r, dict) else "Freed RAM."))

    def _reset_bragg_cal(self) -> None:
        """Reset calibrations on the active file — reload braggpeaks + clear cal state."""
        sc = self._need_active()
        if sc is None or self._busy:
            return
        if QtWidgets.QMessageBox.question(
                self, "Reset calibration",
                f"Reset calibrations for '{sc.name}'?\n\n"
                f"Reloads braggpeaks from disk (uncalibrated) and clears origin / "
                f"ellipse / Q-pixel / basis settings.") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._run_async(lambda: E.reset_bragg_calibration(sc, log=self._console.log),
                        label=f"Reset calibration ({sc.name})",
                        on_done=lambda _r: (self._params.reload(), self._refresh_files(),
                                            self._update_active_views(),
                                            self._status.showMessage(
                                                f"Calibration reset — {sc.name}")))

    def _load_data_and_probe(self) -> None:
        """Detection step: one button to load the datacube + compute the probe so the
        6-point detection runs without going back to Step 1 (Probe)."""
        sc = self._need_active()
        if sc is None or self._busy or not self._path_b_only(sc, "Load data + probe"):
            return
        self._run_async(lambda: E.load_datacube_and_probe(sc, log=self._console.log),
                        label=f"Load data + probe ({sc.name})",
                        on_done=lambda _r: (self._params.set_probe_figure(
                            E.build_probe_figure(sc), focus=False)
                            if E.build_probe_figure(sc) is not None else None,
                            self._update_active_views()))

    def _embed_metadata(self) -> None:
        """Files-dock 'Embed → h5' — write the full analysis metadata into every scan's
        .h5 (so the file is self-describing on reopen / for someone else)."""
        if self._busy or not self._scans:
            return

        def work():
            n = 0
            for sc in self._scans:
                if E.embed_metadata_h5(sc, log=self._console.log):
                    n += 1
            self._console.log(f"Embedded analysis metadata into {n}/{len(self._scans)} file(s).")
            return n

        self._run_async(work, label="Embed metadata → h5",
                        on_done=lambda r: self._status.showMessage(
                            f"Embedded metadata into {r} file(s)." if isinstance(r, int)
                            else "Embedded metadata."))

    def _load_metadata_from_h5(self) -> None:
        """Files-dock — browse an .h5 and import fast4d_metadata into the active scan."""
        sc = self._need_active()
        if sc is None:
            return
        start = str(Path(sc.h5_path or sc.braggpeaks_path or sc.raw_path or "").parent)
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load calibrations from H5 — {sc.name}", start,
            "HDF5 (*.h5 *.hdf5 *.emd);;All (*)")
        if not p:
            return
        meta = E.read_metadata_h5(p)
        if not meta:
            QtWidgets.QMessageBox.warning(
                self, "Metadata",
                f"No fast4d_metadata found in:\n{p}")
            return
        if E.apply_metadata_to_scan(sc, meta):
            if not sc.h5_path:
                sc.h5_path = p
            self._params.reload()
            self._refresh_files()
            self._update_active_views()
            self._console.log(f"[{sc.name}] calibrations restored from {Path(p).name}")
            self._status.showMessage(f"Calibrations loaded from {Path(p).name}")
        else:
            QtWidgets.QMessageBox.warning(
                self, "Metadata", "Could not apply metadata from that file.")

    def _save_session_dialog(self) -> None:
        """Files-dock 'Save…' — always asks where (i.e. Save As)."""
        self._save_results(save_as=True)

    def _save_params_template(self) -> None:
        """Files-dock 'Save params…' — write ONLY the parameter table to a
        Parametros_cal.json (version 3), re-importable via the loader / template."""
        if not self._scans:
            QtWidgets.QMessageBox.information(self, "Save params", "No scans loaded.")
            return
        self._params.apply()                       # commit table edits first
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save parameter table (Parametros_cal.json)",
            "Parametros_cal.json", "JSON (*.json);;All (*)")
        if not p:
            return
        try:
            E.save_params_template(self._scans, p, template_index=max(0, self._active),
                                   log=self._console.log)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save params", f"Could not save:\n{exc}")
            return
        self._status.showMessage(f"Parameter table saved → {p}")

    def _save_results(self, *, save_as: bool) -> None:
        """Save one folder per file (data/ + figures/) + the session JSON.

        Save  → reuse the last chosen location (self._save_root) without asking;
                falls back to asking if none chosen yet.
        Save As → always ask for the location.
        """
        if not self._scans:
            QtWidgets.QMessageBox.information(self, "Save", "Nothing to save yet.")
            return
        d = "" if save_as else (self._save_root or "")
        if not d:
            d = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Save results here (one folder per file: data/ + figures/ + session JSON)")
            if not d:
                return
        dlg = ExportSelectionDialog(self, self._export_categories)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        cats = dlg.result_selection()
        self._export_categories = cats

        self._save_root = d
        self._sync_figure_policy()
        save_dpi = int(self._save_dpi.value())

        def work():
            n = 0
            for sc in self._scans:
                if sc.state is not None and getattr(sc.state, "strain_raw", None):
                    try:
                        E.save_results(sc, output_root=d,
                                       vimg_cmap=self._cmap.currentText(),
                                       log=self._console.log)
                        if cats["per_scan"] and sc.results_dir and (sc.figures or sc.figure_spill):
                            try:
                                E.save_figures(sc, Path(sc.results_dir) / "figures",
                                               dpi=save_dpi, log=self._console.log)
                            except Exception as exc:
                                self._console.log(f"[{sc.name}] figure dump skipped: {exc}")
                        E.clean_duplicate_figure_pngs(sc, log=self._console.log)
                        n += 1
                    except Exception as exc:
                        self._console.log(f"[{sc.name}] save failed: {exc}")
            E.save_session_json(self._scans, str(Path(d) / E.SESSION_FILENAME),
                                log=self._console.log)
            try:                                   # grouped/summarized data + figures
                E.save_summary(self._scans, str(Path(d) / "summary"), log=self._console.log,
                               include_line_figs=cats["lines_group"],
                               include_roi_figs=cats["rois_group"],
                               include_repeatability_figs=cats["repeatability"])
            except Exception as exc:
                self._console.log(f"[warn] summary skipped: {exc}")
            self._console.log(f"Saved {n} result folder(s) + session JSON + summary → {d}")

        self._run_async(work, label="Save results",
                        on_done=lambda _r: self._status.showMessage(f"Saved → {d}"))

    def _export_pptx_report(self) -> None:
        """Report → Export… : PDF / DOCX / PPTX from full panels (no PIL crops)."""
        if not self._scans:
            QtWidgets.QMessageBox.information(self, "Export", "Load files first.")
            return
        # Prefer scans with strain; still allow calib/reports-only export
        ready = list(self._scans)
        has_strain = any(
            (getattr(getattr(sc, "state", None), "strain_raw", None) or {})
            for sc in ready)
        has_figs = any(E.list_figure_keys(sc) for sc in ready)
        if not has_strain and not has_figs:
            QtWidgets.QMessageBox.warning(
                self, "Export",
                "Nothing to export yet.\n\n"
                "Run Compute (for maps) and/or open calibration tools so figures exist.")
            return

        dlg = ExportReportDialog(self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        opts = dlg.result()
        fmt = opts["fmt"]
        if not (opts["include_maps"] or opts["include_calib"] or opts["include_reports"]):
            QtWidgets.QMessageBox.information(
                self, "Export", "Select at least one content category.")
            return
        if opts["include_maps"] and not has_strain:
            QtWidgets.QMessageBox.warning(
                self, "Export",
                "Maps selected but no strain in memory — uncheck Maps or run Compute.")
            return

        filters = {
            "pdf": "PDF (*.pdf)",
            "docx": "Word (*.docx)",
            "pptx": "PowerPoint (*.pptx)",
        }
        default_name = {
            "pdf": "Fast4D_Report.pdf",
            "docx": "Fast4D_Report.docx",
            "pptx": "Fast4D_Report.pptx",
        }[fmt]
        start = Path(self._save_root) if self._save_root else Path.home()
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save report", str(start / default_name),
            f"{filters[fmt]};;All (*)")
        if not out:
            return
        out_path = Path(out)
        template = None
        if fmt == "pptx":
            ans = QtWidgets.QMessageBox.question(
                self, "Export PPTX",
                "Use a PowerPoint template (.pptx)?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.No)
            if ans == QtWidgets.QMessageBox.StandardButton.Cancel:
                return
            if ans == QtWidgets.QMessageBox.StandardButton.Yes:
                p, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Choose PPTX template", "", "PowerPoint (*.pptx);;All (*)")
                if not p:
                    return
                template = Path(p)

        assets_dir = (Path(self._save_root) / "export_assets"
                      if self._save_root else out_path.parent / "export_assets")

        def work():
            from tools.report_export import build_report
            return build_report(
                scans=ready, assets_dir=assets_dir, out_path=out_path,
                fmt=fmt, template=template,
                title="Fast4D Strain / Stress Report",
                include_maps=opts["include_maps"],
                include_calib=opts["include_calib"],
                include_reports=opts["include_reports"],
            )

        def done(res):
            if isinstance(res, Exception):
                QtWidgets.QMessageBox.critical(self, "Export", f"Could not export:\n{res}")
                return
            self._console.log(f"Report exported → {res}")
            self._status.showMessage(f"Report written → {res}")

        self._run_async(work, label=f"Export {fmt.upper()}", on_done=done)

    def _load_session(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load session", "", "Session JSON (*.json);;All (*)")
        if not p:
            return
        try:
            self._scans = E.load_session_json(p)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Session", f"Could not load:\n{exc}")
            return
        self._console.log(f"Session loaded: {len(self._scans)} scan(s) from {Path(p).name}")
        self._active = 0 if self._scans else -1
        self._recent_scan_indices = []
        self._scans_changed()

    def _load_ws(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Load workspace(s) — parent or scan folder")
        if not d:
            return
        root = Path(d)
        subs = [str(c) for c in root.iterdir() if c.is_dir() and (c / "data").is_dir()]
        cands = subs or [str(root)]
        scans = D.hydrate_from_dirs(cands, log=self._console.log, workspace_root=str(root))
        if not scans:
            QtWidgets.QMessageBox.warning(self, "Workspace", "No loadable workspaces here.")
            return
        self._scans = scans
        self._active = 0
        self._recent_scan_indices = []
        jp = E.find_workspace_params_json(root)
        msg = f"Loaded {len(scans)} workspace(s)."
        if jp:
            msg += f" Params: {jp.name}."
        self._console.log(msg)
        self._scans_changed()

    def _refresh_data(self) -> None:
        """Re-hydrate saved-workspace scans from their results dir — sometimes the
        strain (with/without ROI) doesn't land on the state after a workspace load.
        Reloads strain_raw etc. from disk for every scan that has a results_dir."""
        if self._busy:
            return
        todo = [sc for sc in self._scans if getattr(sc, "results_dir", "")]
        if not todo:
            QtWidgets.QMessageBox.information(
                self, "Refresh",
                "No saved-workspace files to reload. Use Load… → 'Add saved "
                "workspace…' first (Refresh re-reads strain from disk).")
            return

        def work():
            for sc in todo:
                dd = Path(sc.results_dir)
                data_dir = dd / "data" if (dd / "data").is_dir() else dd
                try:
                    E.load_results(sc, str(data_dir), log=self._console.log)
                    labels = list((getattr(sc.state, "strain_raw", {}) or {}).keys())
                    self._console.log(f"[{sc.name}] reloaded — strain: {labels or 'none'}")
                except Exception as exc:
                    self._console.log(f"[{sc.name}] reload failed: {exc}")

        self._run_async(work, label="Refresh workspace data",
                        on_done=lambda _r: (self._params.reload(),
                                            self._refresh_files(),
                                            self._update_active_views()))

    def _remove_active(self) -> None:
        if 0 <= self._active < len(self._scans):
            self._scans.pop(self._active)
            self._active = min(self._active, len(self._scans) - 1)
            self._recent_scan_indices = []
            self._scans_changed()

    def _scans_changed(self) -> None:
        if self._active < 0 and self._scans:
            self._active = 0
            self._recent_scan_indices = []
        self._params.rebuild()
        self._params.show_step(self._step)
        self._refresh_files()
        self._update_active_views()
        self._params.report_refresh()
        self._populate_step_actions(self._step)

    def _refresh_files(self) -> None:
        rows = []
        for sc in self._scans:
            icon = _STATUS_ICON.get(sc.status, "○")
            h5 = E.scan_h5_path(sc)
            try:
                size = E.scan_size_info(sc)
            except Exception:
                size = ""
            label = f"{icon} [{D.detect_path(sc)}] {sc.name}"
            if size:
                label += f"   ·   {size}"            # scan size (R / Q) in the Files row
            rows.append({
                "label": label,
                "h5path": h5,
                "tags": [("h5", bool(h5)), ("bragg", bool(sc.braggpeaks_path)),
                         ("par", bool(getattr(sc, "params_source", "")))],
            })
        self._files.refresh(rows, self._active)

    def _on_file_selected(self, row: int) -> None:
        if 0 <= row < len(self._scans):
            self._active = row
            # Delegate the "how many scans stay resident" decision to the
            # configurable ResidentDataPolicy (Settings → Memory) instead of a
            # hardcoded window.
            self._recent_scan_indices = E.enforce_resident_data_limit(
                self._scans, row, self._recent_scan_indices, log=self._console.log)
            # Selection must stay CHEAP: just show the cached/.h5 ADF preview.
            # braggpeaks.h5 is NOT loaded here — that py4DSTEM read is slow and the
            # user only wants to see the ADF. It loads lazily when a calibration
            # tool is opened (_setting_calibration / _apply_calib_step / dialogs
            # all call ensure_braggpeaks_for_calibration themselves).
            self._update_active_views()
            if getattr(self, "_step", None) in ("probe", "select6", "detection"):
                self._populate_step_actions(self._step)

    def _update_active_views(self) -> None:
        """Cheap refresh on selection — calstate + CACHED ADF + probe tab. No disk I/O."""
        sc = self.active_scan()
        self._cal.update_from_state(sc.state if sc else None)
        self._adf.set_image(self._active_adf(load=False), title=(sc.name if sc else "ADF"))
        self._adf.set_line_segments((sc.lines or {}) if sc else {})   # colored + labeled overlay
        self._adf.set_area_roi(E.scan_display_roi(sc) if sc else [])          # calibration ROI fallback
        self._adf.set_area_rois(E.scan_area_rois(sc) if sc else {})           # multi analysis ROIs
        fig = E.build_probe_figure(sc) if sc else None
        if fig is not None:
            self._params.set_probe_figure(fig)
        else:
            self._params.set_probe_images(E.probe_images(sc) if sc else [])
        self._refresh_gallery()
        self._update_workflow_icons()

    def _refresh_gallery(self) -> None:
        items = [(sc.name, E.cached_adf(sc)) for sc in self._scans]
        overlays = {
            i: {"lines": (sc.lines or {}), "roi": E.scan_display_roi(sc)}
            for i, sc in enumerate(self._scans)
        }
        self._gallery.refresh(items, self._active, overlays)

    def _on_gallery_selected(self, idx: int) -> None:
        if 0 <= idx < len(self._scans):
            self._files.select_scan(idx)          # → scanSelected → _on_file_selected

    def _on_node_activated(self, scan_idx: int, h5path: str) -> None:
        """Double-click an h5-root node → read it (light) and show it in the ADF
        viewer. If it's the ADF image, also cache it as the scan's preview."""
        if not (0 <= scan_idx < len(self._scans)):
            return
        sc = self._scans[scan_idx]
        h5file = E.scan_h5_path(sc)
        if not h5file:
            return
        arr = E.read_h5_node(h5file, h5path)
        if arr is None:
            self._console.log(f"[{sc.name}] {h5path} is not a readable array.")
            return
        import numpy as np
        a = np.asarray(arr, dtype=float)
        base = h5path.split("/")[-1]
        if a.ndim != 2:
            self._console.log(f"[{sc.name}] {h5path} shape {a.shape} — not 2-D, not shown.")
            return
        self._active = scan_idx
        if base in E._VIMG_KEYS:                  # cache real virtual images as the preview
            E.set_cached_adf(sc, a)
            self._refresh_gallery()
        self._adf.set_image(a, title=f"{sc.name} · {base}")
        self._console.log(f"[{sc.name}] showing {h5path}  {a.shape} (from h5, no heavy load)")

    def _load_all_adfs(self) -> None:
        if self._busy or not self._scans:
            return
        def work():
            for sc in self._scans:
                if E.cached_adf(sc) is not None:
                    continue
                if not E.scan_h5_path(sc):
                    self._console.log(
                        f"[{sc.name}] no virtual-images .h5 (<stem>.h5) found — can't preview "
                        f"the ADF without it. The raw .mib is NOT loaded for preview (heavy); "
                        f"it loads only when calibration/probe needs it.")
                    continue
                try:
                    E.load_adf(sc, log=self._console.log)   # reads the light .h5, never the .mib
                except Exception as exc:
                    self._console.log(f"[{sc.name}] ADF load failed: {exc}")
        self._run_async(work, label="Load virtual-image previews (.h5)",
                        on_done=lambda _r: self._update_active_views())

    def _active_adf(self, *, load: bool = False):
        """ADF for the active scan. load=False → only what's already in memory."""
        sc = self.active_scan()
        if sc is None:
            return None
        cached = E.cached_adf(sc)
        if cached is not None or not load:
            return cached
        try:
            arr = E.load_adf(sc, log=self._console.log)
            import numpy as np
            return None if arr is None else np.asarray(arr, dtype=float)
        except Exception:
            return None

    def _load_active_adf(self) -> None:
        """Explicit, on-demand ADF load (selection never auto-loads — too slow)."""
        sc = self._need_active()
        if sc is None or self._busy:
            return
        self._run_async(lambda: E.load_adf(sc, log=self._console.log),
                        label=f"Load ADF ({sc.name})",
                        on_done=lambda _r: self._update_active_views())

    # ── picking ──────────────────────────────────────────────────────────────
    def _need_active(self) -> E.Scan | None:
        sc = self.active_scan()
        if sc is None:
            QtWidgets.QMessageBox.information(self, "No scan", "Add or select a scan first.")
        return sc

    def _open_picker(self, *, mode, n_points, title):
        sc = self._need_active()
        if sc is None:
            return None, None
        adf = self._active_adf()
        if adf is None:
            QtWidgets.QMessageBox.warning(
                self, "No image",
                "No ADF for this scan yet. For raw data (Path B) use "
                "'Load data', or type values in the table.")
            return None, None
        dlg = AdfPicker(self, adf, mode=mode, n_points=n_points, title=f"{title} — {sc.name}")
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return sc, dlg.value
        return sc, None

    def _pick_vacuum(self) -> None:
        sc = self._need_active()
        if sc is None or not self._path_b_only(sc, "Choosing a vacuum file"):
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Vacuum scan for probe template", "",
            "Vacuum (*.mib *.dm4 *.h5 *.hdf5);;All (*)")
        if p:
            sc.vacuum_path = p
            self._console.log(f"[{sc.name}] vacuum: {Path(p).name}")
            self._params.reload()

    def _pick_six(self) -> None:
        sc, val = self._open_picker(mode="points", n_points=6, title="Pick 6 Bragg points")
        if val is not None:
            # push the points into the WorkflowState (set_bragg_points) — not just
            # params — so detect_preview / compute_braggpeaks actually see them.
            try:
                E.set_six_points(sc, val, log=self._console.log)
            except Exception as exc:
                self._console.log(f"[{sc.name}] could not set 6 points: {exc}")
                return
            self._console.log(f"[{sc.name}] 6 points set in state: {val}")
            self._params.reload()

    def _pick_roi(self) -> None:
        sc, val = self._open_picker(mode="rect", n_points=2, title="Pick ROI")
        if val is not None:
            sc.params.roi_bounds = [int(v) for v in val]
            self._console.log(f"[{sc.name}] ROI: {sc.params.roi_bounds}")
            self._params.reload()

    def _pick_origin(self) -> None:
        """Pick the diffraction center on the Bragg vector map via the OriginDialog —
        starts at the probe center, live X/Y readout, sampling with explicit Apply
        (and the correct BVM-sampling ↔ diffraction-px scaling)."""
        sc = self._need_active()
        if sc is None:
            return
        self._show_tool(OriginDialog(self))

    def _apply_calib_step(self, step: str) -> None:
        """Apply ONLY this calibration (Origin/Ellipse/Q-pixel/Basis) on top of the
        previous ones — to evaluate a template's calibrations visually, WITHOUT
        reloading the datacube. Needs braggpeaks (loaded light if missing)."""
        sc = self._need_active()
        if sc is None or self._busy:
            return
        self._params.apply()                       # commit table edits first

        def work():
            if getattr(sc.state, "braggpeaks", None) is None:
                E.load_braggpeaks(sc, log=self._console.log)
            E.set_roi(sc, log=self._console.log)   # ROI feeds ellipse/q-pixel use_roi
            # reset to this step's clean PRE-step baseline (or snapshot it the first
            # time) so re-applying doesn't compound on the previous calibration
            E.ensure_pre_step_checkpoint(sc, step, log=self._console.log)
            if step == "origin":
                E.calibrate_origin(sc, log=self._console.log)
            elif step == "ellipse":
                E.calibrate_ellipse(sc, log=self._console.log)
            elif step == "qpixel":
                refit = bool(sc.params.q_refit)
                E.calibrate_q_pixel(sc, refit=refit, make_figure=False, log=self._console.log)
                self._console.log(f"[{sc.name}] applied calibration step: {step}")
                return refit
            elif step == "basis":
                E.calibrate_basis(sc, log=self._console.log)
            else:
                raise ValueError(f"unknown calibration step: {step!r}")
            self._console.log(f"[{sc.name}] applied calibration step: {step}")
            return None

        def on_done(refit_flag):
            if isinstance(refit_flag, Exception):
                return
            if step == "qpixel" and refit_flag is True:
                E.register_q_pixel_fit_figure(sc, log=self._console.log)
            self._params.reload()
            self._update_active_views()
            self._params.report_refresh()

        self._run_async(work, label=f"Apply {step} ({sc.name})",
                        on_done=on_done)

    def _show_tool(self, dlg) -> None:
        """Show a tool window MODELESS so the main window stays usable while it's open
        (e.g. Free RAM, switch files). Keeps a reference (pruning closed ones)."""
        self._tools = [d for d in self._tools if _is_visible(d)]
        self._tools.append(dlg)
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def _open_ellipse_tool(self) -> None:
        """Interactive ellipse calibration window (pick ring → Fit → Apply)."""
        sc = self._need_active()
        if sc is None:
            return
        self._show_tool(EllipseDialog(self))

    def _open_qpixel_tool(self) -> None:
        """Interactive Q-pixel window (Update overlay / Test / Finalize REFIT)."""
        sc = self._need_active()
        if sc is None:
            return
        self._show_tool(QPixelDialog(self))

    def _open_crystal_editor(self) -> None:
        """Define the Q-pixel calibration crystal (element(s) + structure + a) — the
        positions array is generated for the user."""
        sc = self._need_active()
        if sc is None:
            return
        self._show_tool(CrystalEditorDialog(self))

    def _open_roi_tool(self) -> None:
        """Interactive ROI window (drag rectangle / manual bounds, apply to this/all)."""
        sc = self._need_active()
        if sc is None:
            return
        self._show_tool(ROIDialog(self))

    def _open_basis_tuner(self) -> None:
        """Interactive basis window (move basis vars → live choose_basis_vectors preview)."""
        sc = self._need_active()
        if sc is None:
            return
        self._show_tool(BasisDialog(self))

    def _open_virtualization(self) -> None:
        """Open the virtual-images generator (raw datacube → ADF/BF/DP → .h5). The
        dialog has a File selector over the loaded scans; needs ≥1 with a raw 4D path."""
        if not self._scans:
            QtWidgets.QMessageBox.information(self, "Create ADF/BF/DP", "Load files first.")
            return
        if not any(s.raw_path for s in self._scans):
            QtWidgets.QMessageBox.information(
                self, "Create ADF/BF/DP",
                "None of the loaded files has a raw 4D path (.mib/.dm4/.h5) to build from.")
            return
        self._show_tool(VirtualizationDialog(self))

    def _apply_strain_params(self) -> None:
        """Strain step 'Apply': commit the parameter-table edits so the strain params
        (vrange, max_peak_spacing, coordinate_rotation, …) take effect — pushes the
        table values into scan.params (then Compute uses them)."""
        sc = self._need_active()
        if sc is None:
            return
        self._params.apply()                       # commit buffered table edits → params
        self._params.show_step("strain")           # keep the Strain tab visible
        self._console.log(f"[{sc.name}] strain parameters applied: "
                          f"vrange={sc.params.vrange} vrange_theta={sc.params.vrange_theta} "
                          f"max_peak_spacing={sc.params.max_peak_spacing} "
                          f"coordinate_rotation={sc.params.coordinate_rotation}")

    def _apply_full_chain(self, step: str) -> None:
        """'Apply this calibration to this file': run the WHOLE calibration chain from
        scratch on the active file up to AND INCLUDING ``step`` (origin→ellipse→q-pixel
        →basis as needed)."""
        sc = self._need_active()
        if sc is None or self._busy:
            return

        def work():
            E.apply_calibrations_through(sc, step, inclusive=True, log=self._console.log)

        self._run_async(work, label=f"Apply chain → {step} ({sc.name})",
                        on_done=lambda _r: (self._params.reload(),
                                            self._update_active_views()))

    def _reset_calib_step(self, step: str) -> None:
        """Revert the calibration to this step's PRE-step baseline (un-apply this step
        and everything downstream) — so the next test/apply starts clean."""
        sc = self._need_active()
        if sc is None or self._busy:
            return

        def work():
            if getattr(sc.state, "braggpeaks", None) is None:
                E.load_braggpeaks(sc, log=self._console.log)
            E.reset_to_pre_step(sc, step, log=self._console.log)

        self._run_async(work, label=f"Reset {step} ({sc.name})",
                        on_done=lambda _r: (self._params.reload(), self._update_active_views()))

    # ── line tool (Analysis step): same lines across files + per-file drift ──────
    def _set_template(self) -> None:
        sc = self._need_active()
        if sc is None:
            return
        self._template_idx = self._active
        self._console.log(f"Template for lines = '{sc.name}'. Pick lines on it, then Propagate.")

    def _pick_lines(self, targets: list | None = None) -> None:
        if not (0 <= self._template_idx < len(self._scans)):
            self._template_idx = self._active           # default: the active scan
        if not (0 <= self._template_idx < len(self._scans)):
            QtWidgets.QMessageBox.information(self, "Lines", "Add/select a template scan first.")
            return
        tpl = self._scans[self._template_idx]
        adf = E.cached_adf(tpl)
        if adf is None:
            QtWidgets.QMessageBox.warning(self, "Lines", f"No ADF for the template '{tpl.name}'.")
            return
        dlg = AdfPicker(self, adf, mode="lines", n_points=0,
                        title=f"Set up lines on template — {tpl.name}")
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dlg.value:
            return
        self._template_lines = list(dlg.value)          # typed specs (h/v/seg)
        self._console.log(f"Template lines on '{tpl.name}': {len(self._template_lines)} spec(s)")
        self._propagate_lines(use_drift=False, targets=targets)   # scope: this file / all

    def _propagate_lines(self, *, use_drift: bool, targets: list | None = None) -> None:
        if not self._template_lines:
            QtWidgets.QMessageBox.information(self, "Lines", "Set up lines on the template first.")
            return
        pool = targets if targets is not None else self._scans   # None → all files
        pool = [s for s in pool if s is not None]
        if not pool:
            return
        if use_drift and not any(getattr(sc, "drift", None) for sc in pool):
            QtWidgets.QMessageBox.information(
                self, "Lines", "No drift loaded yet — use 'Load drift CSV…' first "
                "(or propagate without drift).")
            return
        E.propagate_template_lines(pool, self._template_lines,
                                   use_drift=use_drift, log=self._console.log)
        self._update_active_views()
        self._console.log(f"Lines propagated to {len(pool)} file(s) "
                          f"({'with' if use_drift else 'no'} drift).")

    def _load_drift_csv(self) -> None:
        if not self._scans:
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load per-file drift CSV (dx, dy vs template)", "", "CSV (*.csv);;All (*)")
        if not p:
            return
        try:
            E.load_drift_csv(p, self._scans, log=self._console.log)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Drift CSV", f"Could not load:\n{exc}")
            return
        if self._template_lines:                        # re-place with the new drift
            self._propagate_lines(use_drift=True)

    def _open_live_line(self) -> None:
        """Menu Tools → Live Line Profile (modeless)."""
        if not self._scans:
            QtWidgets.QMessageBox.information(self, "Live line profile", "Load files first.")
            return
        from qt_measure import MeasureDialog
        self._show_tool(MeasureDialog(self, start="line"))

    def _open_live_roi(self) -> None:
        """Menu Tools → Live ROI profile (modeless)."""
        if not self._scans:
            QtWidgets.QMessageBox.information(self, "Live ROI stats", "Load files first.")
            return
        from qt_measure import MeasureDialog
        self._show_tool(MeasureDialog(self, start="roi"))

    def _popout_adf(self) -> None:
        """ADF viewer '⤢ Open' → the active scan's ADF + lines (colored + labeled) +
        ROI in a separate, resizable window (the matplotlib overlay)."""
        sc = self.active_scan()
        if sc is None or E.cached_adf(sc) is None:
            QtWidgets.QMessageBox.information(self, "ADF", "No ADF loaded for the active file.")
            return
        from qt_widgets import FigureDialog
        self._show_tool(FigureDialog(E.build_lines_overlay_figure(sc), self,
                                     f"ADF — {sc.name}"))

    def _preview_lines(self) -> None:
        sc = self._need_active()
        if sc is None:
            return
        from qt_widgets import FigureDialog
        self._show_tool(FigureDialog(E.build_lines_overlay_figure(sc), self, f"Lines — {sc.name}"))

    def _calculate_lines(self) -> None:
        """Build the line-profile figures (per scan + 'maps with lines') into each
        scan's figures and refresh the Report so the line analysis is ready to view."""
        scans = [s for s in self._scans if (s.lines or {})]
        if not scans:
            QtWidgets.QMessageBox.information(
                self, "Calculate lines",
                "No lines set on any file yet — use 'Set up Lines…' first.")
            return

        def work():
            n = 0
            for sc in scans:
                if not (getattr(sc.state, "strain_raw", None) if sc.state else None):
                    continue
                try:
                    sc.figures["maps_with_lines"] = E.build_maps_with_lines_figure(sc)
                    n += 1
                except Exception as exc:
                    self._console.log(f"[{sc.name}] line figure skipped: {exc}")
            self._console.log(f"Line profiles ready for {n} file(s) — see the Report tab "
                              f"(Line profiles / Lines across files / Maps with lines).")

        self._run_async(work, label="Calculate lines",
                        on_done=lambda _r: self._params.report_refresh())

    def _open_line_setup(self) -> None:
        if not self._scans:
            QtWidgets.QMessageBox.information(self, "Set up Lines", "Load files first.")
            return
        self._show_tool(LineSetupDialog(self))

    def _load_lines_json(self, targets: list | None = None) -> None:
        """Load line definitions from a JSON (the same file used by the data loader)
        and fix them (full-width) on every ADF."""
        if not self._scans:
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load lines from JSON (same JSON as the data loader)", "",
            "JSON (*.json);;All (*)")
        if not p:
            return
        try:
            res = E.load_lines_json(p, self._scans, log=self._console.log)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Lines JSON", f"Could not read:\n{exc}")
            return
        rows = res.get("rows") or []
        if res.get("assigned"):                          # per-scan lines written directly
            # Per-file geometry IS the source of truth (already drift-adjusted per file).
            # Clear any template so a LATER 'Load drift CSV…' does NOT re-propagate generic
            # rows over these real per-scan lines (that was the "two random lines" bug).
            self._template_lines = []
            self._update_active_views()
            self._console.log(
                f"Lines assigned per-scan to {res['assigned']} file(s) from JSON "
                f"(each file its own positions — drift already baked in).")
            return
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Lines JSON",
                "No line definitions found in that JSON (looked for "
                "line_profiles_per_scan / fixed_line_profiles / lines / …).")
            return
        self._template_lines = [{"type": "h", "y": float(y)} for y in rows]
        self._propagate_lines(use_drift=False, targets=targets)
        self._console.log(f"Lines from JSON (template rows): {rows} → propagated.")

    # ── area ROI tool (multiple ROIs; same region across files + per-file drift) ──
    def _pick_area_roi(self, targets: list | None = None, count: int = 1) -> None:
        """Pick ``count`` NEW area ROI rectangles on the active scan's ADF (one picker
        per rectangle) and ADD them (multi-ROI): each gets a fresh id and is merged onto
        the scope (does NOT wipe existing ROIs). Stops early if a picker is cancelled."""
        n = max(1, int(count))
        pool = targets if targets is not None else self._scans
        pool = [s for s in pool if s is not None]
        if not pool:
            return
        added: list = []
        last_bounds = None
        for k in range(n):
            sc, val = self._open_picker(mode="rect", n_points=2,
                                        title=f"Add area ROI ({len(added) + 1}/{n})")
            if val is None:                             # user cancelled this picker
                break
            bounds = [int(v) for v in val]              # [x0,x1,y0,y1]
            rid = E.allocate_roi_ids(self._scans, 1)[0]
            E.place_roi_from_spec(pool, rid, bounds, template_scan=sc, use_drift=False,
                                  log=self._console.log)
            added.append(rid)
            last_bounds = bounds
            self._update_active_views()                 # live feedback per rectangle
        if added:
            self._template_roi = last_bounds            # remember last pick (legacy field)
            self._console.log(f"Added {len(added)} area ROI(s): {', '.join(added)} "
                              f"→ {len(pool)} file(s).")

    def _load_roi_json(self, targets: list | None = None) -> None:
        """Load the area ROI from the SAME JSON used for the lines / data loader."""
        if not self._scans:
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load area ROI from JSON (same JSON as the lines / data loader)", "",
            "JSON (*.json);;All (*)")
        if not p:
            return
        try:
            res = E.load_roi_json(p, self._scans, log=self._console.log)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "ROI JSON", f"Could not read:\n{exc}")
            return
        if res.get("assigned"):                          # per-scan ROIs written directly
            self._template_roi = []                      # don't leave a template that a later
                                                         # 'Propagate ROIs (with drift)' would
                                                         # push over these real per-scan ROIs
            self._update_active_views()
            self._console.log(f"Area ROI assigned per-scan to {res['assigned']} file(s) from JSON.")
            return
        if not res.get("bounds"):
            QtWidgets.QMessageBox.information(
                self, "ROI JSON",
                "No ROI found in that JSON (looked for roi_per_scan / area_roi / "
                "fixed_roi / roi / roi_bounds / profile_roi).")
            return
        self._template_roi = list(res["bounds"])
        pool = targets if targets is not None else self._scans
        pool = [s for s in pool if s is not None]
        if not pool:
            return
        tpl_scan = self.active_scan() or pool[0]
        rid = E.allocate_roi_ids(self._scans, 1)[0]
        E.place_roi_from_spec(pool, rid, list(res["bounds"]), template_scan=tpl_scan,
                              use_drift=False, log=self._console.log)
        self._update_active_views()
        self._console.log(f"Area ROI {rid} from JSON: {res['bounds']} → {len(pool)} file(s).")

    def _propagate_roi(self, *, use_drift: bool, targets: list | None = None) -> None:
        """Propagate the template scan's FULL ROI set onto the scope, each ROI shifted
        by the per-file drift (when requested) and clamped to the image."""
        tpl_scan = self.active_scan() or (self._scans[0] if self._scans else None)
        rois = E.scan_area_rois(tpl_scan) if tpl_scan else {}
        if not rois:
            QtWidgets.QMessageBox.information(
                self, "Area ROIs", "Set up an area ROI first ('Add ROI…' or 'Load ROI JSON…').")
            return
        pool = targets if targets is not None else self._scans
        pool = [s for s in pool if s is not None]
        if not pool:
            return
        if use_drift and not any(getattr(sc, "drift", None) for sc in pool):
            QtWidgets.QMessageBox.information(
                self, "Area ROIs", "No drift loaded yet — use 'Load drift CSV…' first "
                "(the ROI shares the same drift file as the lines).")
            return
        E.propagate_template_rois(pool, rois, use_drift=use_drift, log=self._console.log)
        self._update_active_views()
        self._console.log(f"{len(rois)} area ROI(s) propagated to {len(pool)} file(s) "
                          f"({'with' if use_drift else 'no'} drift).")

    # ── stress maps (Stress step) ────────────────────────────────────────────────
    def _compute_stress(self, *, all_files: bool) -> None:
        """Hooke's-law stress maps from the (already computed) strain — for the active
        scan or all loaded scans. Registers stress_<label> figures for the Report."""
        if self._busy:
            return
        pool = self._scans if all_files else ([self.active_scan()] if self.active_scan() else [])
        targets = [s for s in pool
                   if s is not None and (getattr(s.state, "strain_raw", None) if s.state else None)]
        if not targets:
            QtWidgets.QMessageBox.information(
                self, "Stress",
                "No computed strain found. Load a workspace (or run Compute) first — "
                "stress is derived from the saved strain maps.")
            return

        def work():
            n = 0
            for sc in targets:
                labels = list((getattr(sc.state, "strain_raw", {}) or {}).keys())
                if not labels:
                    continue
                for label in labels:
                    res = E.compute_stress(sc, label=label, mode="plane_stress",
                                           log=self._console.log)
                    if res is not None:
                        n += 1
                if sc.status in ("done", "computed", "pending"):
                    sc.status = "done"
            self._console.log(f"Stress maps computed for {len(targets)} file(s) "
                              f"({n} map(s)). See the Report tab → Per-scan figure → Stress map.")

        self._run_async(work, label="Compute stress maps",
                        on_done=lambda _r: (self._params.reload(),   # reload() also refreshes Report
                                            self._refresh_files(),
                                            self._update_active_views()))

    def _pick_vacuum_region(self) -> None:
        """Pick a vacuum rectangle on the sample's own ADF → probe from that region."""
        sc = self._need_active()
        if sc is None or not self._path_b_only(sc, "Pick vacuum region"):
            return
        sc, val = self._open_picker(mode="rect", n_points=2,
                                    title="Pick vacuum region (sample vacuum)")
        if val is None:
            return
        try:
            E.set_probe_vacuum_roi(sc, [int(v) for v in val], log=self._console.log)
            self._params.reload()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Vacuum region",
                f"Could not set vacuum ROI:\n{exc}\n\n"
                "Load the datacube first ('Load data') — the BF-ROI probe averages "
                "diffraction patterns over this region of the main scan.")

    def _compute_probe(self) -> None:
        """Compute the probe (vacuum file or BF-ROI) → show its images in the Probe tab."""
        sc = self._need_active()
        if sc is None or self._busy or not self._path_b_only(sc, "Compute probe"):
            return
        self._run_async(
            lambda: E.compute_probe(sc, log=self._console.log),
            label=f"compute_probe ({sc.name})",
            on_done=lambda _r: self._params.set_probe_figure(
                E.build_probe_figure(sc), focus=True))

    # ── per-step heavy actions (active scan, Path B tuning) ─────────────────────
    def _async_step(self, fn_name: str) -> None:
        sc = self._need_active()
        if sc is None or self._busy:
            return
        if fn_name in ("load_datacube", "detect_preview") and not self._path_b_only(
                sc, fn_name.replace("_", " ")):
            return
        self._run_async(lambda: getattr(E, fn_name)(sc, log=self._console.log),
                        label=f"{fn_name} ({sc.name})", on_done=lambda _r: self._update_active_views())

    def _open_detect_tuner(self) -> None:
        """Interactive 6-point detection tuner: drag a knob → live re-detect on the
        6 points (mirrors the notebook). Needs datacube + probe loaded; if either
        isn't loaded yet for the active file, this loads them first (the same work
        the "Load data + probe" button does) and THEN opens the tuner automatically
        — no more manual Load-then-Update every time."""
        sc = self._need_active()
        if sc is None:
            return
        if not E.needs_detection_workflow(sc):
            QtWidgets.QMessageBox.information(
                self, "Path A",
                f"'{sc.name}' already has braggpeaks.h5 — the 6-point detection tuner "
                "is only for Path B (before the first braggpeaks file exists).")
            return

        def _show_tuner() -> None:
            from qt_widgets import TunerDialog
            knobs = [
                {"attr": "detect_min_absolute_intensity", "label": "minAbsoluteIntensity (threshold)",
                 "kind": "int", "min": 0, "max": 500, "step": 1},
                {"attr": "detect_min_relative_intensity", "label": "minRelativeIntensity",
                 "kind": "float", "min": 0.0, "max": 1.0, "step": 0.001, "decimals": 3},
                {"attr": "detect_min_peak_spacing", "label": "minPeakSpacing",
                 "kind": "int", "min": 0, "max": 80, "step": 1},
                {"attr": "detect_edge_boundary", "label": "edgeBoundary",
                 "kind": "int", "min": 0, "max": 80, "step": 1},
                {"attr": "detect_sigma", "label": "sigma",
                 "kind": "float", "min": 0.0, "max": 10.0, "step": 0.5, "decimals": 1},
                {"attr": "detect_max_num_peaks", "label": "maxNumPeaks",
                 "kind": "int", "min": 1, "max": 300, "step": 1},
                {"attr": "detect_corr_power", "label": "corrPower",
                 "kind": "float", "min": 0.1, "max": 2.0, "step": 0.05, "decimals": 2},
                {"attr": "detect_subpixel", "label": "subpixel",
                 "kind": "enum", "values": ["none", "poly", "com"]},
                {"attr": "detect_cuda", "label": "CUDA (GPU)", "kind": "bool"},
            ]
            views = [
                {"key": "mode", "label": "View filter", "values": E.DETECT_VIEW_MODES},
                {"key": "cmap", "label": "cmap", "values": E.DETECT_CMAPS},
                {"key": "p_lo", "label": "percentile lo", "kind": "slider",
                 "min": 0.0, "max": 20.0, "step": 0.5, "decimals": 1, "default": 1.0,
                 "depends_on": {"key": "mode", "in": ["pclip_gamma"]}},
                {"key": "p_hi", "label": "percentile hi", "kind": "slider",
                 "min": 80.0, "max": 100.0, "step": 0.1, "decimals": 1, "default": 99.8,
                 "depends_on": {"key": "mode", "in": ["pclip_gamma", "log", "highpass", "raw"]}},
                {"key": "gamma", "label": "gamma", "kind": "slider",
                 "min": 0.1, "max": 2.0, "step": 0.05, "decimals": 2, "default": 0.45,
                 "depends_on": {"key": "mode", "in": ["pclip_gamma"]}},
                {"key": "hp_sigma", "label": "highpass sigma", "kind": "slider",
                 "min": 0.0, "max": 20.0, "step": 0.5, "decimals": 1, "default": 6.0,
                 "depends_on": {"key": "mode", "in": ["highpass"]}},
            ]
            def render(_obj, view):
                return E.build_six_point_detection_figure(
                    sc, view_mode=view.get("mode", "highpass"),
                    cmap=view.get("cmap", "inferno"), log=self._console.log,
                    p_lo=view.get("p_lo", 1.0), p_hi=view.get("p_hi", 99.8),
                    gamma=view.get("gamma", 0.45), hp_sigma=view.get("hp_sigma", 6.0))
            self._show_tool(TunerDialog(
                self, title=f"Tune detection (6 points) — {sc.name}", obj=sc.params,
                knob_specs=knobs, view_specs=views, render_fig=render,
                view={"mode": "highpass", "cmap": "inferno"},
                extra_actions=[("Load data + probe (one click)", self._load_data_and_probe)],
                on_commit=lambda _o: self._params.reload()))

        st = getattr(sc, "state", None)
        needs_load = (getattr(st, "datacube", None) is None
                     or getattr(st, "probe", None) is None)
        if not needs_load:
            _show_tuner()
            return
        if self._busy:
            return
        self._console.log(f"[{sc.name}] datacube/probe not loaded yet — loading "
                          "automatically before opening the detection tuner.")
        self._run_async(
            lambda: E.load_datacube_and_probe(sc, log=self._console.log),
            label=f"Load data + probe ({sc.name})",
            on_done=lambda _r: (
                self._params.set_probe_figure(E.build_probe_figure(sc), focus=False)
                if E.build_probe_figure(sc) is not None else None,
                self._update_active_views(),
                _show_tuner(),
            ))

    def _selected_file_scans(self) -> list:
        """Files-dock multi-selection; falls back to the active file."""
        idxs = self._files.selected_scan_indices()
        if not idxs and 0 <= self._active < len(self._scans):
            idxs = [self._active]
        return [self._scans[i] for i in idxs if 0 <= i < len(self._scans)]

    def _compute_braggpeaks(self) -> None:
        """Queue Path-B detection on the Files-dock selection (Ctrl+click).

        Load data + probe runs automatically per file. Detection parameters come
        from the current (tuned) file. Files that already have braggpeaks.h5 are
        skipped.
        """
        if self._busy:
            return
        self._params.apply()
        selected = self._selected_file_scans()
        if not selected:
            QtWidgets.QMessageBox.information(
                self, "Compute braggpeaks", "Select at least one file in the Files panel.")
            return
        source = self.active_scan() or selected[0]
        path_b = [s for s in selected if E.needs_detection_workflow(s)]
        skipped = [s for s in selected if s not in path_b]
        if not path_b:
            self._path_b_only(selected[0], "Compute braggpeaks")
            return
        src_probe = (source.params.probe_source or "vacuum").lower()
        if src_probe == "vacuum" and not (
                source.vacuum_path and Path(source.vacuum_path).is_file()):
            QtWidgets.QMessageBox.warning(
                self, "Compute braggpeaks",
                f"The current file '{source.name}' has no vacuum file for the probe.\n\n"
                "Set it in the Probe step (Choose vacuum file…) before queuing detection.")
            return
        from qt_splash import ensure_heavy_imports, heavy_ready
        if not heavy_ready():
            self._status.showMessage("Waiting for py4DSTEM to finish loading…")
            ensure_heavy_imports(log=self._console.log, block=True)
        opts = D.ComputeOptions()
        lines = [f"  • {s.name}  →  {D._braggpeaks_save_path(s, opts)}" for s in path_b]
        skip_bit = ""
        if skipped:
            skip_bit = ("\n\nSkipped (already have braggpeaks.h5):\n"
                        + "\n".join(f"  • {s.name}" for s in skipped))
        msg = (
            f"Run full-scan detection for {len(path_b)} file(s)? (heavy)\n"
            f"Detection parameters + vacuum from '{source.name}'.\n"
            "Each file: Load data + probe → braggpeaks → free RAM.\n\n"
            + "\n".join(lines) + skip_bit
        )
        if QtWidgets.QMessageBox.question(self, "Compute braggpeaks", msg) \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        src = source
        pool = list(path_b)
        self._cancel_event.clear()
        self._run_async(
            lambda: D.compute_braggpeaks_all(
                pool, opts, source=src, log=self._console.log,
                progress=self.sig_progress.emit,
                on_scan_done=self.sig_scan_done.emit,
                cancel=self._cancel_event.is_set),
            label=f"braggpeaks ({len(pool)} file(s))",
            on_done=self.sig_finished.emit)

    # ── compute / analysis ─────────────────────────────────────────────────────
    def _compute_options(self) -> D.ComputeOptions:
        self._sync_figure_policy()
        return D.ComputeOptions(
            calibration=self._calib.currentText(),
            do_without_roi=True,
            do_with_roi=self._cb_strain_roi.isChecked(),
            do_stress_without_roi=True,
            do_stress_with_roi=self._cb_stress_roi.isChecked(),
            save=self._cb_save.isChecked(),
            save_figures=self._cb_save.isChecked(),
            figure_mode=self._fig_mode.currentText(),
            max_figures_in_ram=int(self._fig_max.value()),
            close_orphan_pyplot=True,
            store_figure=dict(self._store_figure),
            spill_to_disk=self._cb_spill.isChecked(),
            spill_dpi=int(self._spill_dpi.value()),
            save_figures_dpi=int(self._save_dpi.value()),
            vimg_cmap=self._cmap.currentText())

    def _on_compute(self, targets: list | None = None, *, reuse_save_root: bool = False) -> None:
        """Compute strain. ``targets=None`` → the whole batch (the ▶ Compute button);
        a list → only those scans (e.g. the Strain step's 'Compute this file' after
        fixing ONE file's parameter). ``reuse_save_root`` reuses the last save folder
        (no prompt) so a single-file recompute lands back in the same batch dir."""
        if self._busy:
            return
        from qt_splash import ensure_heavy_imports, heavy_ready
        if not heavy_ready():
            self._status.showMessage("Waiting for py4DSTEM to finish loading…")
            ensure_heavy_imports(log=self._console.log, block=True)
        pool = list(targets) if targets is not None else list(self._scans)
        pool = [s for s in pool if s is not None]
        if not pool:
            QtWidgets.QMessageBox.information(self, "Compute", "Add or select a scan.")
            return
        self._params.apply()                       # commit edits first
        opts = self._compute_options()
        if not (reuse_save_root and self._save_root):
            self._save_root = ""
        if opts.save:
            d = self._save_root
            if not d:
                # ask where to save — one folder per file (data/ + figures/) + session JSON
                d = QtWidgets.QFileDialog.getExistingDirectory(
                    self, "Choose where to save results "
                          "(one folder per file: data/ + figures/ + session JSON)")
                if not d:
                    return                          # cancelled → abort
            opts.output_root = d
            self._save_root = d
        self._cancel_event.clear()
        self._run_async(
            lambda: D.compute_all(pool, opts, log=self._console.log,
                                  progress=self.sig_progress.emit,
                                  on_scan_done=self.sig_scan_done.emit,
                                  cancel=self._cancel_event.is_set),
            label=f"Compute {len(pool)} scan(s)",
            on_done=self.sig_finished.emit)

    def _compute_active(self) -> None:
        """Strain step 'Compute this file': recompute ONLY the selected scan (reusing
        the last save folder so it overwrites that file's results in place)."""
        sc = self._need_active()
        if sc is None:
            return
        self._on_compute(targets=[sc], reuse_save_root=True)

    def _on_cancel(self) -> None:
        if not self._busy:
            return
        self._cancel_event.set()        # cooperative stop at the next step boundary
        self._run_gen += 1              # invalidate the in-flight run → its finish is ignored
        self._set_busy(False)           # free the UI IMMEDIATELY
        self._status.showMessage("Cancelled.")
        self._console.log(
            "Cancel — UI freed immediately. The heavy step already running finishes in "
            "the background (a CUDA call can't be killed mid-flight) and its result is "
            "discarded; no further steps/scans run.")

    def _on_analysis(self, targets: list | None = None) -> None:
        """Analyze (stress + line profiles). ``targets=None`` → all files (the ∑ button);
        a list → only those (e.g. the active file after a single-file recompute)."""
        if self._busy:
            return
        pool = list(targets) if targets is not None else list(self._scans)
        pool = [s for s in pool if s is not None]
        ready = [s for s in pool if (getattr(s.state, "strain_raw", None) if s.state else None)]
        if not ready:
            QtWidgets.QMessageBox.information(
                self, "Analysis", "No computed strain for the selection. Run Compute "
                "(or 'Compute this file') or load a workspace first.")
            return
        opts = D.AnalyzeOptions()
        self._run_async(lambda: D.analyze_all(ready, opts, log=self._console.log),
                        label=f"Analysis ({len(ready)} file(s))", on_done=self.sig_finished.emit)

    def _analyze_active(self) -> None:
        """Analyze (stress + lines) ONLY the selected file."""
        sc = self._need_active()
        if sc is None:
            return
        self._on_analysis(targets=[sc])

    # ── slots (GUI thread) ──────────────────────────────────────────────────────
    @QtCore.Slot(object)
    def _on_progress(self, ev) -> None:
        self._progress.setValue(int((ev.scan_index + 0.5) / max(ev.n_scans, 1) * 100))
        idx = ev.scan_index
        name = getattr(ev, "scan_name", "")
        for i, scn in enumerate(self._scans):
            if scn.name == name:
                idx = i
                break
        if 0 <= idx < len(self._scans) and self._active != idx:
            self._active = idx
            self._refresh_files()
        sc = self.active_scan()
        if sc is not None:
            self._cal.update_from_state(sc.state)
            # Keep ADF/ROI/line overlays synchronized with the scan currently running.
            self._adf.set_image(self._active_adf(load=False), title=sc.name)
            self._adf.set_line_segments(sc.lines or {})
            self._adf.set_area_roi(E.scan_display_roi(sc))
            self._adf.set_area_rois(E.scan_area_rois(sc))
        msg = f"[{ev.scan_index+1}/{ev.n_scans}] {ev.scan_name} · {ev.path} · {ev.step}"
        if ev.message:
            msg += f" — {ev.message}"
        self._status.showMessage(msg)

    @QtCore.Slot(object)
    def _on_scan_done(self, outcome) -> None:
        sc = getattr(outcome, "scan", None)
        if sc is not None:
            try:
                E.flush_deferred_qpixel_figures([sc], log=self._console.log)
            except Exception as exc:
                self._console.log(f"[{sc.name}] deferred Q-pixel figure skipped: {exc}")
            try:
                i = self._scans.index(sc)
                n = max(1, len(self._scans))
                self._progress.setValue(int((i + 1) / n * 100))
            except Exception:
                pass
            self._console.log(
                f"[{sc.name}] GUI updated after file finished "
                f"({'OK' if getattr(outcome, 'ok', False) else 'FAILED'}).")
            try:                                   # warn only: a frame ~90 deg off = mislabelled pair
                for o in E.frame_rotation_outliers(self._scans):
                    if o["name"] == sc.name:
                        self._console.log(
                            f"WARNING [{sc.name}] coordinate_rotation {o['angle_deg']:.2f} deg is "
                            f"{o['diff_deg']:+.1f} deg from the session ({o['session_deg']:.2f} deg, "
                            f"modulo 180) -- check the g1/g2 labels / indexing of this scan.")
            except Exception:
                pass
        # Refresh ONLY this file's column so its calibration/strain/stress images appear
        # as it finishes; a full reload re-rendered every file's thumbnails (5-12 s of
        # frozen GUI per file, growing with the number of files).
        if sc is not None:
            self._params.refresh_scan(sc)
        else:
            self._params.reload()
        self._refresh_files()
        self._update_active_views()
        self._params.report_refresh()

    @QtCore.Slot(object)
    def _on_finished(self, result) -> None:
        self._set_busy(False)
        self._progress.setValue(100)
        n_qfig = E.flush_deferred_qpixel_figures(self._scans, log=self._console.log)
        if n_qfig:
            self._console.log(f"Built {n_qfig} deferred Q-pixel FIT figure(s) on GUI thread.")
        self._params.reload()                      # fitted q_px, etc.
        self._refresh_files()
        self._update_active_views()
        self._params.report_refresh()                     # new figures are now available
        if isinstance(result, Exception):
            self._status.showMessage("Failed — see console.")
        elif isinstance(result, D.BatchOutcome):
            labels = [lab for o in result.outcomes
                      for lab in (getattr(o, "labels", None) or [])]
            kind = "Braggpeaks" if "braggpeaks" in labels else "Compute"
            if getattr(self, "_save_root", ""):
                try:
                    p = E.save_session_json(
                        self._scans, str(Path(self._save_root) / E.SESSION_FILENAME),
                        log=self._console.log)
                    self._console.log(f"Session saved → {p}")
                except Exception as exc:
                    self._console.log(f"[warn] could not write session JSON: {exc}")
                try:                               # grouped/summarized data + figures
                    E.save_summary(self._scans, str(Path(self._save_root) / "summary"),
                                   log=self._console.log)
                except Exception as exc:
                    self._console.log(f"[warn] summary skipped: {exc}")
            self._status.showMessage(f"{kind} done: {result.summary()}")
        elif isinstance(result, dict):
            n = sum(len(v.get("stress", {})) for v in result.values())
            self._status.showMessage(f"Analysis done: stress on {n} map(s).")
        else:
            self._status.showMessage("Done.")

    # ── threading ────────────────────────────────────────────────────────────
    def _run_async(self, work, *, label: str, on_done=None) -> None:
        self._set_busy(True, label)
        self._run_gen += 1
        gen = self._run_gen             # this run's id; a Cancel bumps _run_gen → stale

        def runner():
            result = None
            try:
                result = work()
            except Exception as exc:
                self._console.log(f"[ERROR] {label}: {exc}\n{traceback.format_exc()}")
                result = exc

            def finish(r=result, g=gen):
                if g != self._run_gen:          # cancelled/superseded → discard, don't touch UI
                    self._console.log(f"[{label}] background result discarded (cancelled).")
                    return
                self._set_busy(False, label)
                if on_done is not None:
                    on_done(r)
                self._maybe_tidy_figures()

            # marshal completion to the GUI thread via a queued signal (the worker
            # thread has no event loop, so QTimer.singleShot here would never fire)
            self.sig_call.emit(finish)

        threading.Thread(target=runner, daemon=True, name=label).start()

    def _set_busy(self, busy: bool, label: str = "") -> None:
        self._busy = busy
        self._btn_compute.setEnabled(not busy)
        self._btn_reset_cal.setEnabled(not busy)
        self._btn_cancel.setEnabled(busy)
        if busy:
            self._progress.setValue(0)
            self._status.showMessage(f"Running: {label} …")
