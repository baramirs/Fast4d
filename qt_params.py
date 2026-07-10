"""fast4d.qt_params — the Excel-like calibration parameter table (PySide6).

A ``QTabWidget`` with one tab per calibration step + ROI (the user's request:
"tabla tipo Excel, con pestañas para cada calibración y ROI"). Each tab is a
``QTableWidget`` whose rows are parameters and columns are the loaded files
(single == one column). This is the single editable home of every scan's
calibration values; ``engine.run_calibration_sequence`` reads them back.

Edits buffer in the cells and commit to ``scan.params`` only on **Apply**
(explicit, per the agreed design). Bool → checkbox cell; enum → combobox cell;
read-only (fitted px, picked points) → greyed, non-editable.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

import engine as E
from param_spec import (PARAM_GROUPS, ParamSpec, format_value, group_for_step,
                        group_specs, parse_value)
from qt_widgets import ClickableFigureLabel, ProbeView

_READONLY_BG = QtGui.QColor("#F0F0F0")
_READONLY_FG = QtGui.QColor("#777777")
_HDR_FG = QtGui.QColor("#0D47A1")

# Which figure(s) (scan.figures keys) to show below each file's column, per tab.
# A str = one figure row; a list = several rows (the Stress tab shows BOTH the
# no-ROI and ROI stress maps under each file, like the other per-step figures).
GROUP_FIGURE_KEY = {
    "detection": "detection", "roi": None, "origin": "origin",
    "ellipse": "ellipse", "qpixel": "q_pixel", "basis": "basis",
    "strain": ["strain_without_roi", "strain_with_roi"],
    "tools": ["stress_without_roi", "stress_with_roi"],
}
# Row labels for the figure rows (the vertical-header text).
FIG_ROW_LABEL = {
    "detection": "Detection", "origin": "Origin (+res)", "ellipse": "Ellipse",
    "q_pixel": "Q-pixel", "basis": "Basis",
    "strain_without_roi": "Strain", "strain_with_roi": "Strain (ROI)",
    "stress_without_roi": "Stress", "stress_with_roi": "Stress (ROI)",
}
# Detection-tab cells skipped on Path A (braggpeaks already on disk).
_PATH_A_DET_CELL = "(Path A — braggpeaks)"
_PATH_A_DET_TIP = (
    "Path A: braggpeaks.h5 loaded — vacuum / probe source / 6 points are not used "
    "for calibration. Path B columns stay editable.")


class _GroupTable(QtWidgets.QTableWidget):
    """One tab: rows = a group's params, columns = scans. Buffered edits."""

    # Emits the clicked column's scan index, so the host window can keep the
    # Files panel (the "who does Play Calibration/Analysis act on" selection)
    # in sync with whatever file the user is pointing at in this table.
    fileClicked = QtCore.Signal(int)

    def __init__(self, group_key: str, get_scans, parent=None) -> None:
        super().__init__(parent)
        self._group_key = group_key
        self._get_scans = get_scans
        self._rows: list[tuple[str, ParamSpec]] = group_specs(group_key)
        fk = GROUP_FIGURE_KEY.get(group_key)
        self._fig_keys: list[str] = ([] if fk is None
                                     else ([fk] if isinstance(fk, str) else list(fk)))
        self.verticalHeader().setDefaultSectionSize(26)
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setMinimumSectionSize(90)
        self.setAlternatingRowColors(True)
        self.setMinimumWidth(260)
        self._fill_origin: tuple | None = None      # (row, col) where a fill-drag began
        self._fill_value = None                     # that cell's value, copied while dragging
        self.build()

    # ── build / reload ──────────────────────────────────────────────────────
    def build(self) -> None:
        scans = self._get_scans() or []
        self.clear()
        n_param = len(self._rows)
        n_fig = len(self._fig_keys)
        self.setRowCount(n_param + n_fig)
        self.setColumnCount(max(len(scans), 1))
        labels = [spec.label for _sk, spec in self._rows]
        for fk in self._fig_keys:
            labels.append(FIG_ROW_LABEL.get(fk, "Figure"))
        self.setVerticalHeaderLabels(labels)
        if scans:
            self.setHorizontalHeaderLabels([sc.name[:22] for sc in scans])
            for c in range(len(scans)):
                self.horizontalHeaderItem(c).setForeground(_HDR_FG)
        else:
            self.setHorizontalHeaderLabels(["(no files)"])

        for r, (sk, spec) in enumerate(self._rows):
            for c, sc in enumerate(scans):
                self._make_cell(r, c, sc, spec, sk)
            if not scans:
                self.setItem(r, 0, self._readonly_item(""))

        if self._fig_keys and scans:                 # calibration/result figure(s) per file column
            for fi, fk in enumerate(self._fig_keys):
                row = n_param + fi
                self.setRowHeight(row, 152)
                for c, sc in enumerate(scans):
                    fig = (sc.figures or {}).get(fk)
                    spill = E.resolve_figure_path(sc, fk) if not fig else ""
                    self.setCellWidget(row, c, ClickableFigureLabel(
                        fig, spill_path=spill, title=f"{sc.name} — {fk}",
                        scan=sc, fig_key=fk))

    def reload(self) -> None:
        self.build()

    # ── cells ──────────────────────────────────────────────────────────────
    def _readonly_item(self, text: str) -> QtWidgets.QTableWidgetItem:
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)   # not editable/selectable-edit
        it.setBackground(_READONLY_BG)
        it.setForeground(_READONLY_FG)
        return it

    def _path_a_skip_detection(self, scan, step_key: str) -> bool:
        """Probe / 6-pt rows are Path-B-only; per-column when scans differ."""
        return (self._group_key == "detection"
                and step_key in ("probe", "select6")
                and E.analysis_path(scan) == "A")

    def _scan_value(self, scan, spec: ParamSpec):
        if spec.kind == "scan_path":
            return getattr(scan, spec.field, "") or ""
        return getattr(scan.params, spec.field, None)

    def _make_cell(self, r: int, c: int, scan, spec: ParamSpec, step_key: str) -> None:
        if self._path_a_skip_detection(scan, step_key):
            it = self._readonly_item(_PATH_A_DET_CELL)
            it.setToolTip(_PATH_A_DET_TIP)
            self.setItem(r, c, it)
            return
        val = self._scan_value(scan, spec)
        if spec.kind == "bool":
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable
                        | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Checked if bool(val)
                             else QtCore.Qt.CheckState.Unchecked)
            it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.setItem(r, c, it)
        elif spec.kind == "enum":
            combo = QtWidgets.QComboBox()
            combo.addItems(list(spec.options))
            cur = str(val) if val is not None else (spec.options[0] if spec.options else "")
            i = combo.findText(cur)
            combo.setCurrentIndex(i if i >= 0 else 0)
            self.setCellWidget(r, c, combo)
        elif spec.readonly or spec.kind in ("fitted", "points", "scan_path"):
            it = self._readonly_item(format_value(spec, val))
            if spec.kind == "scan_path" and val:
                it.setToolTip(str(val))
            self.setItem(r, c, it)
        else:
            it = QtWidgets.QTableWidgetItem(format_value(spec, val))
            it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.setItem(r, c, it)

    # ── commit / copy ─────────────────────────────────────────────────────────
    def commit(self) -> list[str]:
        """Write editable cells into each scan.params. Returns error strings."""
        scans = self._get_scans() or []
        errors: list[str] = []
        for r, (_sk, spec) in enumerate(self._rows):
            if spec.readonly or spec.kind in ("fitted", "points", "scan_path"):
                continue
            for c, sc in enumerate(scans):
                if self._path_a_skip_detection(sc, _sk):
                    continue
                try:
                    if spec.kind == "bool":
                        item = self.item(r, c)
                        new = item.checkState() == QtCore.Qt.CheckState.Checked
                    elif spec.kind == "enum":
                        new = self.cellWidget(r, c).currentText()
                    else:
                        new = parse_value(spec, self.item(r, c).text())
                except (ValueError, TypeError, AttributeError) as exc:
                    errors.append(f"{sc.name}/{spec.label}: {exc}")
                    continue
                setattr(sc.params, spec.field, new)
        return errors

    def copy_first_to_all(self) -> None:
        scans = self._get_scans() or []
        if len(scans) < 2:
            return
        for r, (_sk, spec) in enumerate(self._rows):
            if spec.readonly or spec.kind in ("fitted", "points", "scan_path"):
                continue
            if self._path_a_skip_detection(scans[0], _sk):
                continue
            if spec.kind == "enum":
                src = self.cellWidget(r, 0).currentText()
                for c in range(1, len(scans)):
                    if self._path_a_skip_detection(scans[c], _sk):
                        continue
                    w = self.cellWidget(r, c)
                    j = w.findText(src)
                    if j >= 0:
                        w.setCurrentIndex(j)
            elif spec.kind == "bool":
                src = self.item(r, 0).checkState()
                for c in range(1, len(scans)):
                    if self._path_a_skip_detection(scans[c], _sk):
                        continue
                    self.item(r, c).setCheckState(src)
            else:
                src = self.item(r, 0).text()
                for c in range(1, len(scans)):
                    if self._path_a_skip_detection(scans[c], _sk):
                        continue
                    self.item(r, c).setText(src)

    # ── horizontal drag-to-fill (Excel-like): drag a cell left/right → copy its
    #    value across that ROW; vertical drag does nothing ─────────────────────────
    def _capture(self, r: int, c: int):
        if not (0 <= r < len(self._rows)):
            return None
        spec = self._rows[r][1]
        if spec.kind == "bool":
            it = self.item(r, c); return it.checkState() if it else None
        if spec.kind == "enum":
            w = self.cellWidget(r, c); return w.currentText() if w else None
        it = self.item(r, c); return it.text() if it else None

    def _apply(self, r: int, c: int, val) -> None:
        if not (0 <= r < len(self._rows)):
            return
        sk, spec = self._rows[r]
        scans = self._get_scans() or []
        if c < len(scans) and self._path_a_skip_detection(scans[c], sk):
            return
        if spec.readonly or spec.kind in ("fitted", "points", "scan_path"):
            return
        if spec.kind == "bool":
            it = self.item(r, c)
            if it is not None and val is not None:
                it.setCheckState(val)
        elif spec.kind == "enum":
            w = self.cellWidget(r, c)
            if w is not None and val is not None:
                j = w.findText(str(val))
                if j >= 0:
                    w.setCurrentIndex(j)
        else:
            it = self.item(r, c)
            if it is not None and val is not None:
                it.setText(str(val))

    def _fill_row(self, r: int, c_from: int, c_to: int) -> None:
        """Copy (r, c_from)'s value to every cell between c_from and c_to in row r."""
        val = self._capture(r, c_from)
        if val is None:
            return
        lo, hi = sorted((int(c_from), int(c_to)))
        for c in range(lo, hi + 1):
            if c != c_from:
                self._apply(r, c, val)

    def mousePressEvent(self, e) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            idx = self.indexAt(e.position().toPoint() if hasattr(e, "position")
                               else e.pos())
            self._fill_origin = (idx.row(), idx.column()) if idx.isValid() else None
            if idx.isValid():
                self.fileClicked.emit(idx.column())
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if (e.buttons() & QtCore.Qt.MouseButton.LeftButton) and self._fill_origin:
            idx = self.indexAt(e.position().toPoint() if hasattr(e, "position") else e.pos())
            r0, c0 = self._fill_origin
            if idx.isValid() and idx.row() == r0 and idx.column() != c0 \
                    and 0 <= r0 < len(self._rows):
                self._fill_row(r0, c0, idx.column())     # horizontal → fill the row
                return                                   # suppress the selection drag
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        self._fill_origin = None
        super().mouseReleaseEvent(e)


class ParamTable(QtWidgets.QWidget):
    """QTabWidget of _GroupTable + Apply / Copy / Reload bar.

    ``get_scans()`` returns the live list of engine.Scan objects.
    ``applied`` fires (with the error list, empty == clean) after Apply.
    """

    applied = QtCore.Signal(list)
    tabStep = QtCore.Signal(str)        # emitted when the USER selects a tab → icon strip syncs
    fileSelected = QtCore.Signal(int)   # user clicked a file column → host syncs the Files panel

    def __init__(self, get_scans, parent=None) -> None:
        super().__init__(parent)
        self._get_scans = get_scans
        self._tables: dict[str, _GroupTable] = {}
        self._report = None             # persistent Report tab (created on first rebuild)
        self._suppress_tab = False      # guard: don't echo programmatic tab changes back

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setMovable(True)               # tabs reorderable too
        self._tabs.currentChanged.connect(self._on_tab_changed)
        lay.addWidget(self._tabs, 1)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(0, 2, 0, 0)
        bar.setSpacing(6)
        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet("color:#2E7D32;")
        self._status.setMinimumWidth(220)
        bar.addWidget(self._status, 0)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setMinimumWidth(420)
        self._progress.setFixedHeight(18)
        self._progress.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                     QtWidgets.QSizePolicy.Policy.Fixed)
        self._progress.setStyleSheet(
            "QProgressBar{border:1px solid #2E7D32; border-radius:4px; "
            "background:#E8F5E9; text-align:center; color:#1B5E20;}"
            "QProgressBar::chunk{background:#43A047; border-radius:3px;}")
        bar.addWidget(self._progress, 1)
        b_apply = QtWidgets.QPushButton("Apply")
        b_apply.clicked.connect(self.apply)
        b_copy = QtWidgets.QPushButton("Copy file 1 → all")
        b_copy.clicked.connect(self._copy_current)
        b_reload = QtWidgets.QPushButton("Reload")
        b_reload.clicked.connect(self.reload)
        for b in (b_copy, b_reload, b_apply):
            bar.addWidget(b)
        self._bar = bar
        lay.addLayout(bar)

        self.rebuild()

    # ── structure ────────────────────────────────────────────────────────────
    def rebuild(self) -> None:
        """(Re)create the tabs — call when the scan list changes. The Report tab is
        persistent (kept across rebuilds) and always sits LAST, after Stress."""
        # detach the persistent Report so clear() doesn't drop it
        if self._report is not None:
            idx = self._tabs.indexOf(self._report)
            if idx >= 0:
                self._tabs.removeTab(idx)
            self._report.setParent(None)
        self._tabs.clear()
        self._tables.clear()
        # Probe results tab (populated by the probe, NOT by the file's params).
        self._probe_view = ProbeView()
        self._tabs.addTab(self._probe_view, "Probe")
        for gk, title, _steps in PARAM_GROUPS:
            tbl = _GroupTable(gk, self._get_scans)
            tbl.fileClicked.connect(self.fileSelected)
            self._tables[gk] = tbl
            self._tabs.addTab(tbl, title)
        # Report — the LAST tab (user choice: in the calibration window, after Stress)
        if self._report is None:
            from qt_report import ReportPanel
            self._report = ReportPanel(self._get_scans)
        self._tabs.addTab(self._report, "Report")
        self._report.refresh()

    def set_probe_images(self, images: list, *, focus: bool = False) -> None:
        """Show the active scan's probe output images in the Probe tab."""
        pv = getattr(self, "_probe_view", None)
        if pv is not None:
            pv.set_images(images)
            if focus:
                self._tabs.setCurrentWidget(pv)

    def set_probe_figure(self, fig, *, focus: bool = False) -> None:
        """Show a full probe Figure (the 4-panel notebook view) in the Probe tab."""
        pv = getattr(self, "_probe_view", None)
        if pv is not None:
            pv.set_figure(fig)
            if focus:
                self._tabs.setCurrentWidget(pv)

    def reload(self) -> None:
        for tbl in self._tables.values():
            tbl.reload()
        self.report_refresh()
        self._status.setText("Reloaded from params.")

    @property
    def report(self):
        return self._report

    @property
    def progress_bar(self):
        return self._progress

    def add_progress_widget(self, widget: QtWidgets.QWidget) -> None:
        """Place an external action beside the green progress/loading bar."""
        self._bar.insertWidget(2, widget)

    def report_refresh(self) -> None:
        if self._report is not None:
            self._report.refresh()

    def show_step(self, step_key: str) -> None:
        """Select the tab for *step_key* (icon-strip nav).

        The Probe *step* has its own results tab (``ProbeView``), separate from the
        Detection param table — only ``probe`` lands there; select6/detection still
        open the Detection table."""
        if step_key == "probe":
            pv = getattr(self, "_probe_view", None)
            if pv is not None:
                self._suppress_tab = True
                self._tabs.setCurrentWidget(pv)
                self._suppress_tab = False
            return
        gk = group_for_step(step_key)
        if gk is None:
            return
        tbl = self._tables.get(gk)
        if tbl is None:
            return
        self._suppress_tab = True
        self._tabs.setCurrentWidget(tbl)
        self._suppress_tab = False

    def _on_tab_changed(self, *_a) -> None:
        """User picked a tab → tell the icon strip to select the matching step."""
        if self._suppress_tab:
            return
        w = self._tabs.currentWidget()
        if w is getattr(self, "_probe_view", None):
            self.tabStep.emit("probe")
            return
        for gk, tbl in self._tables.items():         # group key == icon step key
            if tbl is w:
                self.tabStep.emit(gk)
                return

    # ── actions ──────────────────────────────────────────────────────────────
    def apply(self) -> list[str]:
        errors: list[str] = []
        for tbl in self._tables.values():
            errors.extend(tbl.commit())
        n = len(self._get_scans() or [])
        if errors:
            self._status.setStyleSheet("color:#C62828;")
            self._status.setText("⚠ " + "; ".join(errors[:2]))
        else:
            self._status.setStyleSheet("color:#2E7D32;")
            self._status.setText(f"Applied to {n} file(s) ✓")
        self.applied.emit(errors)
        return errors

    def _copy_current(self) -> None:
        w = self._tabs.currentWidget()
        if isinstance(w, _GroupTable):
            w.copy_first_to_all()
            self._status.setStyleSheet("color:#1565C0;")
            self._status.setText("Copied file 1 → all (press Apply).")
