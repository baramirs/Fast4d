"""fast4d.qt_report — the Report panel: the user chooses what to view.

A single dockable panel that browses everything the session produced, on demand:

  • Per-scan figure   — any registered ``scan.figures`` entry (probe, origin with
                        residuals, ellipse, q-pixel overlay, basis, strain/stress
                        maps, 6-point detection…), embedded with a pan/zoom toolbar.
  • Cross-scan analysis (``analysis.py``, light — reads saved strain arrays):
        Strain distribution (hist+KDE) · Box/Violin by channel · Channel
        correlation (per scan) · PCA of scans · Stress summary (table) · Strain
        stats (table).

Figures show as a clickable thumbnail (click, or "Maximize", opens a full
pan/zoom dialog); tables render in a QTableWidget and can be exported to
CSV/XLSX. Analysis is computed synchronously (it's light) under a wait cursor.

The panel auto-renders whenever the user changes View / Scan / Item / Map
(or after Refresh / workspace load). Only the *current* selection is built —
never every map at once — so RAM stays bounded.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

import engine as E

# friendly labels for the per-scan figure keys (engine.FIGURE_ORDER + extras)
FIG_LABELS = {
    "probe": "Probe (4-panel)", "select6": "Detection @ 6 points",
    "detection": "Detection", "roi": "ROI",
    "origin": "Origin (+ residuals)", "ellipse": "Ellipse",
    "q_pixel": "Q-pixel calibration (fit)", "basis": "Basis vectors",
    "strain_without_roi": "Strain map (no ROI)", "strain_with_roi": "Strain map (ROI ref)",
    "stress_without_roi": "Stress map (no ROI)", "stress_with_roi": "Stress map (ROI ref)",
    "line_profiles": "Line profiles (per scan)", "maps_with_lines": "Maps with lines",
    "roi_profiles": "ROI stats (per scan)", "maps_with_rois": "Maps with ROIs",
    "roi_distribution": "ROI distribution (per scan)",
}

# view kinds
K_FIG = "Per-scan figure"
K_LINE_PROF = "Line profiles (per scan)"
K_LINE_GROUP = "Lines across files (grouped)"
K_LINE_TABLE = "Line stats across files (table)"
K_DIST = "Strain distribution (all scans)"
K_BOX = "Box/Violin by channel (all scans)"
K_CORR = "Channel correlation (per scan)"
K_PCA = "PCA of scans (all scans)"
K_STRESS = "Stress summary (table)"
K_STATS = "Strain stats (table)"
K_PIXDIFF = "Pixel-wise difference (repeatability)"
K_PIXDIFF_TABLE = "Repeatability metrics (table)"
K_MAPS_LINES = "Maps with lines (per scan)"
K_ROI_PROF = "ROI stats (per scan)"
K_ROI_DIST = "ROI value distribution (per scan)"
K_ROI_GROUP = "ROIs across files (grouped)"
K_ROI_TABLE = "ROI stats across files (table)"
K_MAPS_ROIS = "Maps with ROIs (per scan)"
K_CALIB_PLOT = "Calibration values across files"
K_CALIB_TABLE = "Calibration values (table)"
CATEGORIES = [
    ("Per-scan figures",    [K_FIG]),
    ("Lines / Profiles",    [K_LINE_PROF, K_MAPS_LINES, K_LINE_GROUP, K_LINE_TABLE]),
    ("ROIs",                [K_ROI_PROF, K_ROI_DIST, K_MAPS_ROIS, K_ROI_GROUP, K_ROI_TABLE]),
    ("Calibration",         [K_CALIB_PLOT, K_CALIB_TABLE]),
    ("Cross-scan analysis", [K_DIST, K_BOX, K_CORR, K_PCA, K_STRESS, K_STATS]),
    ("Repeatability",       [K_PIXDIFF, K_PIXDIFF_TABLE]),
]
_TABLE_KINDS = {K_STRESS, K_STATS, K_LINE_TABLE, K_ROI_TABLE, K_PIXDIFF_TABLE, K_CALIB_TABLE}
_PERSCAN_KINDS = {K_FIG, K_CORR, K_LINE_PROF, K_MAPS_LINES, K_ROI_PROF, K_ROI_DIST, K_MAPS_ROIS}
_GROUPLINE_KINDS = {K_LINE_GROUP, K_LINE_TABLE}
_GROUPROI_KINDS = {K_ROI_GROUP, K_ROI_TABLE}
_PIXDIFF_KINDS = {K_PIXDIFF, K_PIXDIFF_TABLE}
# Views that silently combine EVERY currently loaded scan unless the user has
# explicitly opted into "Repro. exp." (engine.AnalysisScopePolicy.shared_stats).
# K_PIXDIFF/K_PIXDIFF_TABLE are deliberately excluded — the user always picks
# the two files by hand there, so there's no silent mixing to guard against.
_GROUPED_ANALYSIS_KINDS = {K_DIST, K_BOX, K_PCA, K_STRESS, K_STATS} | _GROUPLINE_KINDS | _GROUPROI_KINDS
_CHANNELS = [("ε_yy", "eyy"), ("ε_xx", "exx"), ("ε_xy", "exy"),
             ("σ_xx", "sxx"), ("σ_yy", "syy"), ("σ_xy", "sxy"), ("ADF", "adf")]


class _ViewSelector(QtWidgets.QWidget):
    """One category's "View:" combo + its scan/ref/item/map filter controls."""

    changed = QtCore.Signal()

    def __init__(self, kinds: list[str], get_scans, parent=None) -> None:
        super().__init__(parent)
        self._get_scans = get_scans
        self._quiet = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._kind_row = QtWidgets.QWidget()
        row1 = QtWidgets.QHBoxLayout(self._kind_row); row1.setContentsMargins(0, 0, 0, 0)
        self._kind = QtWidgets.QComboBox(); self._kind.addItems(kinds)
        self._kind.currentIndexChanged.connect(self._on_kind)
        row1.addWidget(QtWidgets.QLabel("View:")); row1.addWidget(self._kind, 1)
        lay.addWidget(self._kind_row)
        if len(kinds) == 1:
            self._kind_row.setVisible(False)   # nothing to choose

        row2 = QtWidgets.QHBoxLayout()
        self._scan2 = QtWidgets.QComboBox()        # reference scan (pixel-diff: A)
        self._lbl_scan2 = QtWidgets.QLabel("Ref:")
        self._scan = QtWidgets.QComboBox()
        self._scan.currentIndexChanged.connect(self._on_scan)
        self._scan2.currentIndexChanged.connect(self._on_scan2)
        self._item = QtWidgets.QComboBox()
        self._item.currentIndexChanged.connect(self._notify)
        self._label = QtWidgets.QComboBox(); self._label.addItems(["without_roi", "with_roi"])
        self._label.currentIndexChanged.connect(self._notify)
        self._lbl_scan = QtWidgets.QLabel("Scan:")
        self._lbl_item = QtWidgets.QLabel("Item:")
        self._lbl_label = QtWidgets.QLabel("Map:")
        for w in (self._lbl_scan2, self._scan2, self._lbl_scan, self._scan,
                  self._lbl_item, self._item, self._lbl_label, self._label):
            row2.addWidget(w)
        row2.addStretch(1)
        lay.addLayout(row2)

        row2b = QtWidgets.QHBoxLayout()
        self._chk_drift_corr = QtWidgets.QCheckBox("Corregir por drift")
        self._chk_drift_corr.setToolTip(
            "Desplaza B (sub-píxel) al marco de A usando el drift cargado "
            "('Load drift CSV…') antes de calcular Δ.")
        self._chk_drift_corr.toggled.connect(self._notify)
        self._lbl_drift_info = QtWidgets.QLabel("")
        self._lbl_drift_info.setStyleSheet("color:#666; font-size:10px;")
        row2b.addWidget(self._chk_drift_corr)
        row2b.addWidget(self._lbl_drift_info)
        row2b.addStretch(1)
        lay.addLayout(row2b)

        self._on_kind()

    def refresh(self) -> None:
        self._quiet = True
        try:
            self._on_kind()
        finally:
            self._quiet = False

    def _notify(self, *_args) -> None:
        if not self._quiet:
            self.changed.emit()

    def _all_line_ids(self) -> list:
        ids: list = []
        for sc in (self._get_scans() or []):
            for lid in (getattr(sc, "lines", None) or {}):
                if lid not in ids:
                    ids.append(lid)
        return ids

    def _all_roi_ids(self) -> list:
        ids: list = []
        for sc in (self._get_scans() or []):
            for rid in E.scan_area_rois(sc):
                if rid not in ids:
                    ids.append(rid)
        return ids

    def _on_kind(self) -> None:
        kind = self._kind.currentText()
        scans = self._get_scans() or []
        per_scan = kind in _PERSCAN_KINDS
        group_line = kind in _GROUPLINE_KINDS
        group_roi = kind in _GROUPROI_KINDS
        pixdiff = kind in _PIXDIFF_KINDS
        has_item = kind in (K_FIG, K_DIST, K_BOX, K_LINE_PROF, K_LINE_GROUP, K_LINE_TABLE,
                            K_ROI_PROF, K_ROI_DIST, K_ROI_GROUP, K_ROI_TABLE,
                            K_CALIB_PLOT,
                            K_PIXDIFF, K_PIXDIFF_TABLE)
        has_label = kind not in (K_FIG, K_CALIB_PLOT, K_CALIB_TABLE)
        # main scan/line combo: K_PIXDIFF needs "Scan B"; K_PIXDIFF_TABLE doesn't (all vs Ref)
        show_scan = per_scan or group_line or group_roi or (kind == K_PIXDIFF)
        self._lbl_scan.setVisible(show_scan); self._scan.setVisible(show_scan)
        self._lbl_scan.setText("ROI:" if group_roi
                               else ("Line:" if group_line
                                     else ("Scan B:" if kind == K_PIXDIFF else "Scan:")))
        self._lbl_scan2.setVisible(pixdiff); self._scan2.setVisible(pixdiff)
        self._lbl_scan2.setText("Ref (A):")
        self._chk_drift_corr.setVisible(pixdiff)
        self._lbl_drift_info.setVisible(pixdiff)
        self._lbl_item.setVisible(has_item); self._item.setVisible(has_item)
        self._lbl_label.setVisible(has_label); self._label.setVisible(has_label)

        # scan / line combo
        cur = self._scan.currentIndex()
        self._scan.blockSignals(True); self._scan.clear()
        if group_line:
            for lid in self._all_line_ids():
                self._scan.addItem(lid)
        elif group_roi:
            for rid in self._all_roi_ids():
                self._scan.addItem(rid)
        else:
            for sc in scans:
                self._scan.addItem(sc.name)
        if 0 <= cur < self._scan.count():
            self._scan.setCurrentIndex(cur)
        self._scan.blockSignals(False)
        # reference-scan combo (pixel diff)
        cur2 = self._scan2.currentIndex()
        self._scan2.blockSignals(True); self._scan2.clear()
        for sc in scans:
            self._scan2.addItem(sc.name)
        self._scan2.setCurrentIndex(cur2 if 0 <= cur2 < self._scan2.count() else 0)
        self._scan2.blockSignals(False)

        # item combo
        self._item.blockSignals(True); self._item.clear()
        if kind == K_FIG:
            self._populate_fig_items()
        elif kind in (K_LINE_PROF, K_LINE_GROUP, K_LINE_TABLE,
                      K_ROI_PROF, K_ROI_DIST, K_ROI_GROUP, K_ROI_TABLE,
                      K_PIXDIFF, K_PIXDIFF_TABLE):
            for lbl, val in _CHANNELS:
                self._item.addItem(lbl, val)
        elif kind == K_CALIB_PLOT:
            cols = E.calibration_numeric_columns(scans)
            for col in cols:
                self._item.addItem(col, col)
            if not cols:
                self._item.addItem("(no numeric calibration values yet)", None)
        elif kind == K_DIST:
            for ch in ("eyy", "exx", "exy"):
                self._item.addItem(ch, ch)
        elif kind == K_BOX:
            self._item.addItem("violin", "violin"); self._item.addItem("box", "box")
        self._item.blockSignals(False)
        self._update_drift_info()
        self._notify()

    def _on_scan(self) -> None:
        if self._kind.currentText() == K_FIG:
            self._item.blockSignals(True); self._item.clear()
            self._populate_fig_items(); self._item.blockSignals(False)
        self._update_drift_info()
        self._notify()

    def _on_scan2(self) -> None:
        self._update_drift_info()
        self._notify()

    def _update_drift_info(self) -> None:
        if self._kind.currentText() not in _PIXDIFF_KINDS:
            return
        scans = self._get_scans() or []
        ia = self._scan2.currentIndex()
        if not (0 <= ia < len(scans)):
            self._lbl_drift_info.setText("")
            self._chk_drift_corr.setEnabled(False)
            return
        da = getattr(scans[ia], "drift", None)
        if self._kind.currentText() == K_PIXDIFF:
            ib = self._scan.currentIndex()
            db = getattr(scans[ib], "drift", None) if 0 <= ib < len(scans) else None
            if da and db:
                ddx, ddy = db[0] - da[0], db[1] - da[1]
                self._lbl_drift_info.setText(
                    f"A drift=({da[0]:+.1f},{da[1]:+.1f})  B drift=({db[0]:+.1f},{db[1]:+.1f})  "
                    f"Δ(B−A)=({ddx:+.1f},{ddy:+.1f}) px")
                self._chk_drift_corr.setEnabled(True)
            else:
                self._lbl_drift_info.setText("(drift no cargado — usa 'Load drift CSV…')")
                self._chk_drift_corr.setEnabled(False)
        else:  # K_PIXDIFF_TABLE
            if da:
                self._lbl_drift_info.setText(f"Ref (A) drift=({da[0]:+.1f},{da[1]:+.1f})")
                self._chk_drift_corr.setEnabled(True)
            else:
                self._lbl_drift_info.setText("(drift no cargado — usa 'Load drift CSV…')")
                self._chk_drift_corr.setEnabled(False)

    def _populate_fig_items(self) -> None:
        scans = self._get_scans() or []
        i = self._scan.currentIndex()
        if not (0 <= i < len(scans)):
            return
        figs = E.collect_figures(scans[i])      # ordered {key: Figure}
        for key in figs:
            if key.startswith("line_group_"):
                lbl = f"Grouped line {key[11:]}"
            elif key.startswith("roi_group_"):
                lbl = f"Grouped ROI {key[10:]}"
            else:
                lbl = FIG_LABELS.get(key, key)
            self._item.addItem(lbl, key)
        if not figs:
            self._item.addItem("(no figures yet — run Compute)", None)


class ReportPanel(QtWidgets.QWidget):
    saveRequested = QtCore.Signal(bool)     # True → "Save As" (always ask); False → "Save"
    exportPptxRequested = QtCore.Signal()   # build PPTX report from saved summary/images

    def __init__(self, get_scans, get_active_scan=None, parent=None) -> None:
        super().__init__(parent)
        self._get_scans = get_scans
        self._get_active_scan = get_active_scan
        self._fig = None            # currently shown Figure (for Maximize)
        self._df = None             # currently shown DataFrame (for Export)
        self._canvas = None         # ClickableFigureLabel thumbnail (figure page)
        self._suppress_auto = False

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # ── controls: one tab per view category ─────────────────────────────────
        self._tabs = QtWidgets.QTabWidget()
        self._selectors: list[_ViewSelector] = []
        for label, kinds in CATEGORIES:
            sel = _ViewSelector(kinds, self._get_scans)
            sel.changed.connect(self._on_selector_changed)
            self._selectors.append(sel)
            self._tabs.addTab(sel, label)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        lay.addWidget(self._tabs)

        row3 = QtWidgets.QHBoxLayout()
        for txt, fn, tip in (
            ("Show", self._show,
             "Re-render the current selection. Usually unnecessary — the view "
             "updates automatically when you change View / Scan / Item / Map, "
             "or after loading data / Compute."),
            ("Maximize", self._maximize, None),
            ("Export table…", self._export, None),
            ("Export PPTX…", lambda: self.exportPptxRequested.emit(), None),
            ("Save", lambda: self.saveRequested.emit(False), None),
            ("Save As…", lambda: self.saveRequested.emit(True), None),
        ):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            if tip:
                b.setToolTip(tip)
            row3.addWidget(b)
        row3.addStretch(1)
        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet("color:#1565C0; font-size:10px;")
        row3.addWidget(self._status)
        lay.addLayout(row3)

        # ── display: stacked (figure page / table page) ─────────────────────────
        self._stack = QtWidgets.QStackedWidget()
        figpage = QtWidgets.QWidget(); self._fig_host = QtWidgets.QVBoxLayout(figpage)
        self._fig_host.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QtWidgets.QLabel(
            "Select a view — it shows automatically.\n"
            "(Per-scan figures appear after Compute or when a workspace is loaded; "
            "analysis needs computed strain.)")
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color:#888;")
        self._fig_host.addWidget(self._placeholder, 1)
        self._table = QtWidgets.QTableWidget()
        self._stack.addWidget(figpage)        # index 0
        self._stack.addWidget(self._table)    # index 1
        lay.addWidget(self._stack, 1)

        self.refresh()

    # ── population ──────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Rebuild selector contents, then auto-show the current selection.

        Only the *current* view is rendered (lazy). We deliberately do not
        pre-build every figure/map into the display — that would spike RAM
        without helping UX. Workspace hydrate already owns figure objects;
        Show just references the selected one.
        """
        self._suppress_auto = True
        try:
            for sel in self._selectors:
                sel.refresh()
        finally:
            self._suppress_auto = False
        self._auto_show()

    def _on_selector_changed(self) -> None:
        if self._suppress_auto:
            return
        if self.sender() is not self._tabs.currentWidget():
            return
        self._auto_show()

    def _on_tab_changed(self, _index: int = 0) -> None:
        if self._suppress_auto:
            return
        self._auto_show()

    def _auto_show(self) -> None:
        scans = self._get_scans() or []
        if not scans:
            self._status.setText("No scans loaded.")
            self._set_figure(None)
            return
        self._show()

    # ── render ────────────────────────────────────────────────────────────────
    def _show(self) -> None:
        scans = self._get_scans() or []
        if not scans:
            self._status.setText("No scans loaded."); return
        sel = self._tabs.currentWidget()
        kind = sel._kind.currentText()
        label = sel._label.currentText()
        restricted_to_active = False
        if kind in _GROUPED_ANALYSIS_KINDS and not E.get_analysis_scope().shared_stats:
            active = self._get_active_scan() if self._get_active_scan else None
            if active is not None:
                scans = [active]
                restricted_to_active = True
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            import analysis as A
            self._df = None
            if kind == K_FIG:
                i = sel._scan.currentIndex()
                key = sel._item.currentData()
                fig = E.collect_figures(scans[i]).get(key) if (0 <= i < len(scans)) else None
                if fig is None:
                    self._status.setText("No figure for that selection."); return
                self._set_figure(fig)
            elif kind == K_DIST:
                self._set_figure(A.distribution_figure(scans, channel=sel._item.currentData(),
                                                       label=label))
            elif kind == K_BOX:
                self._set_figure(A.boxplot_figure(scans, label=label,
                                                  kind=sel._item.currentData()))
            elif kind == K_LINE_PROF:
                i = sel._scan.currentIndex()
                if not (0 <= i < len(scans)):
                    self._status.setText("Pick a scan."); return
                self._set_figure(E.build_line_profiles_figure(
                    scans[i], label, sel._item.currentData()))
            elif kind == K_LINE_GROUP:
                lid = sel._scan.currentText()
                if not lid:
                    self._status.setText("No lines set — use Analysis → Set up Lines & ROI."); return
                self._set_figure(E.build_grouped_line_figure(
                    scans, lid, sel._item.currentData(), label))
            elif kind == K_LINE_TABLE:
                lid = sel._scan.currentText()
                if not lid:
                    self._status.setText("No lines set — use Analysis → Set up Lines & ROI."); return
                self._set_table(E.grouped_line_table(
                    scans, lid, sel._item.currentData(), label))
            elif kind == K_MAPS_LINES:
                i = sel._scan.currentIndex()
                if not (0 <= i < len(scans)):
                    self._status.setText("Pick a scan."); return
                self._set_figure(E.build_maps_with_lines_figure(scans[i], label))
            elif kind == K_ROI_PROF:
                i = sel._scan.currentIndex()
                if not (0 <= i < len(scans)):
                    self._status.setText("Pick a scan."); return
                self._set_figure(E.build_roi_profiles_figure(
                    scans[i], label, sel._item.currentData()))
            elif kind == K_ROI_DIST:
                i = sel._scan.currentIndex()
                if not (0 <= i < len(scans)):
                    self._status.setText("Pick a scan."); return
                self._set_figure(E.build_roi_distribution_figure(
                    scans[i], label, sel._item.currentData()))
            elif kind == K_ROI_GROUP:
                rid = sel._scan.currentText()
                if not rid:
                    self._status.setText("No ROIs set — use Analysis → Set up Lines & ROI → Area ROIs."); return
                self._set_figure(E.build_grouped_roi_figure(
                    scans, rid, sel._item.currentData(), label))
            elif kind == K_ROI_TABLE:
                rid = sel._scan.currentText()
                if not rid:
                    self._status.setText("No ROIs set — use Analysis → Set up Lines & ROI → Area ROIs."); return
                self._set_table(E.grouped_roi_table(
                    scans, rid, sel._item.currentData(), label))
            elif kind == K_MAPS_ROIS:
                i = sel._scan.currentIndex()
                if not (0 <= i < len(scans)):
                    self._status.setText("Pick a scan."); return
                self._set_figure(E.build_maps_with_rois_figure(scans[i], label))
            elif kind == K_CALIB_PLOT:
                value = sel._item.currentData()
                self._set_figure(E.build_calibration_value_figure(scans, value))
            elif kind == K_CALIB_TABLE:
                self._set_table(E.calibration_values_table(scans))
            elif kind == K_CORR:
                i = sel._scan.currentIndex()
                self._set_figure(A.correlation_figure(scans[i], label))
            elif kind == K_PCA:
                self._set_figure(A.pca_figure(scans, label))
            elif kind == K_STRESS:
                c11, c12, c44 = scans[0].params.stress_constants_gpa()
                self._set_table(A.stress_summary_table(
                    scans, c11_gpa=c11, c12_gpa=c12, c44_gpa=c44, label=label))
            elif kind == K_STATS:
                self._set_table(A.cross_scan_stats(scans, label))
            elif kind == K_PIXDIFF:
                ia, ib = sel._scan2.currentIndex(), sel._scan.currentIndex()
                if not (0 <= ia < len(scans) and 0 <= ib < len(scans)):
                    self._status.setText("Pick Ref (A) and Scan B."); return
                if ia == ib:
                    self._status.setText("Pick two DIFFERENT scans (A ≠ B)."); return
                self._set_figure(E.build_pixel_difference_figure(
                    scans[ia], scans[ib], sel._item.currentData(), label,
                    drift_correct=sel._chk_drift_corr.isChecked()))
            elif kind == K_PIXDIFF_TABLE:
                ref = sel._scan2.currentIndex()
                self._set_table(E.pixel_difference_table(
                    scans, sel._item.currentData(), label, reference_idx=max(0, ref),
                    drift_correct=sel._chk_drift_corr.isChecked()))
            if restricted_to_active:
                self._status.setText(
                    f"Showing: {kind} — active file only (enable 'Repro. exp.' in "
                    "Analysis to compare across files)")
            else:
                self._status.setText(f"Showing: {kind}")
        except Exception as exc:
            self._status.setText(f"Error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _set_figure(self, fig) -> None:
        # Show a clickable thumbnail (no embedded toolbar/canvas churn); click
        # opens the same FigureDialog as the "Maximize" button for full pan/zoom.
        self._fig = fig
        self._stack.setCurrentIndex(0)
        if self._canvas is not None:
            self._fig_host.removeWidget(self._canvas)
            self._canvas.setParent(None); self._canvas.deleteLater(); self._canvas = None
        if fig is None:
            self._placeholder.setVisible(True)
            return
        self._placeholder.setVisible(False)
        from qt_widgets import ClickableFigureLabel
        # Cap the thumbnail to the figure page's actual available size so a large
        # composite figure (e.g. origin + residuals, 6 panels) can't force the
        # whole window past the screen — this QLabel has no scroll fallback.
        avail_w, avail_h = self._stack.width(), self._stack.height()
        max_w = min(1100, avail_w - 16) if avail_w > 316 else 1100
        max_h = min(820, avail_h - 16) if avail_h > 236 else 820
        self._canvas = ClickableFigureLabel(
            fig, title="Report figure", max_w=max_w, max_h=max_h, dpi=110)
        self._fig_host.addWidget(self._canvas, 1)

    def _set_table(self, df) -> None:
        self._df = df
        self._fig = None
        t = self._table
        t.clear()
        cols = list(df.columns) if df is not None else []
        t.setColumnCount(len(cols)); t.setHorizontalHeaderLabels([str(c) for c in cols])
        t.setRowCount(0 if df is None else len(df))
        if df is not None:
            for r in range(len(df)):
                for c, col in enumerate(cols):
                    it = QtWidgets.QTableWidgetItem(str(df.iloc[r, c]))
                    it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
                    t.setItem(r, c, it)
        t.resizeColumnsToContents()
        self._stack.setCurrentIndex(1)

    def _maximize(self) -> None:
        if self._fig is None:
            self._status.setText("Maximize works on a figure view."); return
        from qt_widgets import FigureDialog, _is_visible_dialog
        host = self.window()
        if host is not None and not hasattr(host, "_figure_windows"):
            host._figure_windows = []
        if host is not None:
            host._figure_windows = [d for d in host._figure_windows if _is_visible_dialog(d)]
        dlg = FigureDialog(self._fig, host, "Report figure")
        if host is not None:
            host._figure_windows.append(dlg)
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def _export(self) -> None:
        if self._df is None:
            self._status.setText("Export works on a table view."); return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export table", "", "CSV (*.csv);;Excel (*.xlsx)")
        if not p:
            return
        try:
            import analysis as A
            A.export_dataframe(self._df, p)
            self._status.setText(f"Exported → {p}")
        except Exception as exc:
            self._status.setText(f"Export failed: {exc}")
