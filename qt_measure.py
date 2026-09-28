"""fast4d.qt_measure -- one tool for Line and ROI profiles (replaces Live Line / Live ROI).

* + Line / + ROI; ids L1, L2… / R1, R2…; one colour per element.
* Select by clicking the element on the map or its name in the list; the Delete
  key (or "Delete selected") removes ONLY that element; "Clear all" asks first.
* A line shows a BAR at its start and an ARROWHEAD at its end; the profile
  distance runs from the start. Width 1..200 px (yellow band on the map).
* ROIs can be moved, resized and ROTATED (handle on the right side).
* The anchored properties panel edits start/end/length/width (line) or
  centre/length/width/angle (ROI), in nm when the template file has a scan step,
  px otherwise.
* Line mode plots the profile of every line on every checked file; ROI mode plots
  mean ± SD per file and the histogram of the selected ROI. Ctrl+C copies the
  plotted data as tab-separated text with headers and units.
* The drift of every file is re-read on each update (and polled), so a drift
  loaded from the drift plugin shows up without reopening the tool. Elements move
  onto file s by -(drift_s - drift_template).

Numbers come from measure_model (Qt-free, tested).
"""
from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import engine as E
import measure_model as MM

_CHANS = [("ε_xx", "exx"), ("ε_yy", "eyy"), ("ε_xy", "exy"), ("θ", "theta"),
          ("σ_xx", "sxx"), ("σ_yy", "syy"), ("σ_xy", "sxy"), ("ADF", "adf")]


def _value_unit(label_text: str) -> str:
    """'ε_yy (%)' → '%'; 'ADF' → 'a.u.'."""
    if "(" in label_text and label_text.endswith(")"):
        return label_text[label_text.rfind("(") + 1:-1]
    return "a.u."


def _poly(points) -> QtGui.QPolygonF:
    return QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in points])


class MeasureDialog(QtWidgets.QDialog):
    """Line / ROI profiles on a template map, sampled on every checked file."""

    def __init__(self, host, start: str = "line") -> None:
        super().__init__(host)
        from qt_main import _apply_channel_map_item, _attach_live_colorbar, _enable_minmax
        import pyqtgraph as pg
        _enable_minmax(self)
        self._pg = pg
        self._paint_map = _apply_channel_map_item
        self._host = host
        self._scans = list(host._scans)
        self._model = MM.MeasureModel()
        self._items: dict = {}               # id → {"roi", "deco": [...], "label"}
        self._map_shape: tuple | None = None
        self._syncing = False
        self._loaded_ids: set = set()
        self._last_curves: list = []
        self._last_rows: list = []
        self._drift_sig = MM.drift_signature(self._scans)
        self.setWindowTitle("Line / ROI profiles")
        self.resize(1320, 800)
        self._build(pg, _attach_live_colorbar)
        self._reload_map()
        self._load_elements_from_template()
        if not self._model.elements and self._map_shape is not None:
            (self._add_roi if start == "roi" else self._add_line)()
        self._upd_timer = QtCore.QTimer(self)
        self._upd_timer.setSingleShot(True)
        self._upd_timer.setInterval(60)
        self._upd_timer.timeout.connect(self._update_plots)
        self._drift_timer = QtCore.QTimer(self)
        self._drift_timer.setInterval(1500)
        self._drift_timer.timeout.connect(self._poll_drift)
        self._drift_timer.start()
        self._sync_panel()
        self._update_plots()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self, pg, attach_cbar) -> None:
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Template file:"))
        self._file = QtWidgets.QComboBox()
        for s in self._scans:
            self._file.addItem(s.name)
        if 0 <= getattr(self._host, "_active", 0) < len(self._scans):
            self._file.setCurrentIndex(self._host._active)
        self._file.currentIndexChanged.connect(self._on_template_changed)
        top.addWidget(self._file, 1)
        top.addWidget(QtWidgets.QLabel("Map:"))
        self._chan = QtWidgets.QComboBox()
        for lbl, val in _CHANS:
            self._chan.addItem(lbl, val)
        self._chan.setCurrentIndex(1)                      # ε_yy, as before
        self._chan.currentIndexChanged.connect(self._reload_map)
        top.addWidget(self._chan)
        top.addWidget(QtWidgets.QLabel("Ref:"))
        self._label = QtWidgets.QComboBox()
        for key, title in E.ROI_REF_LABELS.items():
            self._label.addItem(title, key)
        self._label.currentIndexChanged.connect(self._reload_map)
        top.addWidget(self._label)
        top.addWidget(QtWidgets.QLabel("R:"))
        self._r_px = QtWidgets.QDoubleSpinBox()
        self._r_px.setRange(0.0, 1_000_000.0)
        self._r_px.setDecimals(3)
        self._r_px.setSuffix(" pm/px")
        self._r_px.setToolTip("Scan step of the template file (pm per pixel). "
                              "0 = distances in pixels; > 0 = distances in nm.")
        self._r_px.valueChanged.connect(self._on_r_px_changed)
        top.addWidget(self._r_px)
        for txt, fn, tip in (
            ("+ Line", lambda: self._add_line(), "Add a line (bar = start, arrow = end)."),
            ("+ ROI", lambda: self._add_roi(), "Add a rectangle ROI (rotatable)."),
            ("Delete selected", self._delete_selected, "Delete only the selected element (Del)."),
            ("Clear all", self._clear_all, "Delete every element (asks first)."),
        ):
            b = QtWidgets.QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            top.addWidget(b)
        lay.addLayout(top)

        split = QtWidgets.QSplitter()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._glw = pg.GraphicsLayoutWidget()
        self._vb = self._glw.addViewBox(row=0, col=0)
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)
        self._img = pg.ImageItem()
        self._vb.addItem(self._img)
        self._cbar = attach_cbar(self._glw, pg)
        split.addWidget(self._glw)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self._plots = QtWidgets.QStackedWidget()
        self._prof = pg.PlotWidget()
        self._prof.addLegend(offset=(10, 10))
        self._prof.showGrid(x=True, y=True, alpha=0.3)
        self._plots.addWidget(self._prof)
        roi_page = QtWidgets.QWidget()
        rp = QtWidgets.QVBoxLayout(roi_page)
        rp.setContentsMargins(0, 0, 0, 0)
        self._means = pg.PlotWidget()
        self._means.addLegend(offset=(10, 10))
        self._means.showGrid(x=True, y=True, alpha=0.3)
        self._hist = pg.PlotWidget()
        self._hist.addLegend(offset=(10, 10))
        self._hist.setLabel("left", "counts")
        rp.addWidget(self._means, 1)
        rp.addWidget(self._hist, 1)
        self._plots.addWidget(roi_page)
        rl.addWidget(self._plots, 1)
        rl.addWidget(self._build_panel())
        split.addWidget(right)
        split.setSizes([560, 760])
        lay.addWidget(split, 1)

        bottom = QtWidgets.QHBoxLayout()
        files_box = QtWidgets.QGroupBox("Files to plot / send to Report")
        fl = QtWidgets.QHBoxLayout(files_box)
        self._file_list = QtWidgets.QListWidget()
        self._file_list.setMaximumHeight(90)
        ti = self._file.currentIndex()
        for i, sc in enumerate(self._scans):
            it = QtWidgets.QListWidgetItem(sc.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if i == ti
                             else QtCore.Qt.CheckState.Unchecked)
            self._file_list.addItem(it)
        self._file_list.itemChanged.connect(lambda *_: self._schedule())
        fl.addWidget(self._file_list, 1)
        side = QtWidgets.QVBoxLayout()
        for txt, fn in (("All", lambda: self._check_files(lambda i: True)),
                        ("None", lambda: self._check_files(lambda i: False)),
                        ("Template only",
                         lambda: self._check_files(lambda i: i == self._file.currentIndex()))):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            side.addWidget(b)
        fl.addLayout(side)
        bottom.addWidget(files_box, 1)
        dv = QtWidgets.QVBoxLayout()
        self._chk_drift = QtWidgets.QCheckBox("Apply drift")
        self._chk_drift.setToolTip("Move each element onto file s by -(drift_s - drift_template). "
                                   "Checked automatically when a drift is loaded.")
        self._chk_drift.setChecked(MM.any_drift(self._scans))
        self._chk_drift.toggled.connect(lambda *_: self._schedule())
        dv.addWidget(self._chk_drift)
        b_drift = QtWidgets.QPushButton("Load drift CSV…")
        b_drift.clicked.connect(self._load_drift)
        dv.addWidget(b_drift)
        dv.addStretch(1)
        bottom.addLayout(dv)
        lay.addLayout(bottom)

        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox()
        b_rep = bb.addButton("Send to Report", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        b_rep.setToolTip("Save the elements (ids as shown) on the checked files and build "
                         "the Report figures.")
        b_rep.clicked.connect(self._send_to_report)
        bb.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        lay.addWidget(bb)

        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Copy, self, activated=self._copy)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Delete, self,
                        activated=self._delete_selected)

    def _build_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Elements")
        box.setFixedWidth(250)
        v = QtWidgets.QVBoxLayout(box)
        self._elist = QtWidgets.QListWidget()
        self._elist.setMaximumHeight(130)
        self._elist.currentRowChanged.connect(self._on_list_pick)
        v.addWidget(self._elist)
        self._sel_title = QtWidgets.QLabel("")
        self._sel_title.setStyleSheet("font-weight:bold;")
        v.addWidget(self._sel_title)
        self._forms = QtWidgets.QStackedWidget()

        def dsb(lo=-1e6, hi=1e6, dec=3):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._on_prop_edit)
            return s

        lf = QtWidgets.QWidget()
        lform = QtWidgets.QFormLayout(lf)
        self._lx0, self._ly0, self._lx1, self._ly1 = dsb(), dsb(), dsb(), dsb()
        self._llen = dsb(0.0)
        self._lw = QtWidgets.QSpinBox()
        self._lw.setRange(1, E.LINE_WIDTH_MAX)
        self._lw.setSuffix(" px")
        self._lw.setKeyboardTracking(False)
        self._lw.setToolTip(f"Integration width, 1-{E.LINE_WIDTH_MAX} px (yellow band).")
        self._lw.valueChanged.connect(self._on_prop_edit)
        self._line_labels = []
        for name, w in (("start x", self._lx0), ("start y", self._ly0),
                        ("end x", self._lx1), ("end y", self._ly1), ("length", self._llen)):
            lab = QtWidgets.QLabel(name)
            self._line_labels.append((lab, name))
            lform.addRow(lab, w)
        lform.addRow("width", self._lw)
        self._forms.addWidget(lf)

        rf = QtWidgets.QWidget()
        rform = QtWidgets.QFormLayout(rf)
        self._rcx, self._rcy = dsb(), dsb()
        self._rlen, self._rwid = dsb(0.0), dsb(0.0)
        self._rang = dsb(-360.0, 360.0, 2)
        self._rang.setSuffix(" °")
        self._roi_labels = []
        for name, w in (("centre x", self._rcx), ("centre y", self._rcy),
                        ("length", self._rlen), ("width", self._rwid)):
            lab = QtWidgets.QLabel(name)
            self._roi_labels.append((lab, name))
            rform.addRow(lab, w)
        rform.addRow("angle", self._rang)
        self._forms.addWidget(rf)
        self._forms.addWidget(QtWidgets.QLabel("No element selected."))
        v.addWidget(self._forms)
        hint = QtWidgets.QLabel("Line: bar = start, arrow = end.\nDel deletes the selected "
                                "element.\nCtrl+C copies the plotted data.")
        hint.setStyleSheet("color:#666; font-size:10px;")
        v.addWidget(hint)
        v.addStretch(1)
        return box

    # ── helpers ───────────────────────────────────────────────────────────────
    def _tpl(self):
        i = self._file.currentIndex()
        return self._scans[i] if 0 <= i < len(self._scans) else None

    def _checked_scans(self) -> list:
        return [self._scans[i] for i in range(self._file_list.count())
                if self._file_list.item(i).checkState() == QtCore.Qt.CheckState.Checked]

    def _check_files(self, pred) -> None:
        for i in range(self._file_list.count()):
            self._file_list.item(i).setCheckState(
                QtCore.Qt.CheckState.Checked if pred(i) else QtCore.Qt.CheckState.Unchecked)

    def _scale(self) -> float:
        return MM.nm_per_px(self._tpl()) or 1.0

    def _unit(self) -> str:
        return MM.unit_of(self._tpl())

    def _schedule(self, *_):
        if hasattr(self, "_upd_timer"):
            self._upd_timer.start()

    # ── map ───────────────────────────────────────────────────────────────────
    def _on_template_changed(self, *_):
        tpl = self._tpl()
        self._r_px.blockSignals(True)
        self._r_px.setValue(float(getattr(getattr(tpl, "params", None), "r_px_pm", 0.0) or 0.0))
        self._r_px.blockSignals(False)
        self._reload_map()
        self._sync_panel()

    def _reload_map(self, *_):
        tpl = self._tpl()
        if tpl is not None and self._r_px.value() == 0.0:
            self._r_px.blockSignals(True)
            self._r_px.setValue(float(getattr(tpl.params, "r_px_pm", 0.0) or 0.0))
            self._r_px.blockSignals(False)
        ch, lab = self._chan.currentData(), self._label.currentData() or "without_roi"
        m = E.channel_map_2d(tpl, ch, lab) if tpl is not None else None
        if m is None:
            self._img.clear()
            self._status.setText(f"No '{ch}' map for this file (compute strain"
                                 + ("/stress" if str(ch).startswith("s") else "") + " first).")
            return
        a = np.asarray(m, dtype=float)
        self._paint_map(self._img, a, tpl, ch, pg_mod=self._pg, cbar=self._cbar)
        first = self._map_shape is None
        self._map_shape = a.shape
        if first:
            self._vb.autoRange()
        self._schedule()

    def _on_r_px_changed(self, *_):
        tpl = self._tpl()
        if tpl is None:
            return
        tpl.params.r_px_pm = float(self._r_px.value())
        try:
            E.apply_r_pixel_from_params(tpl)
        except Exception:
            pass
        self._sync_panel()
        self._schedule()

    # ── elements ↔ graphics ───────────────────────────────────────────────────
    def _load_elements_from_template(self) -> None:
        tpl = self._tpl()
        if tpl is None or self._map_shape is None:
            return
        widths = getattr(tpl, "line_widths", None) or {}
        for lid, seg in sorted((getattr(tpl, "lines", None) or {}).items()):
            el = self._model.add_line(self._map_shape, seg, widths.get(lid, 3), eid=lid)
            self._make_items(el)
            self._loaded_ids.add(lid)
        for rid, b in sorted(E.scan_area_rois(tpl).items()):
            el = self._model.add_roi(self._map_shape, b, eid=rid)
            self._make_items(el)
            self._loaded_ids.add(rid)
        self._refresh_list()

    def _add_line(self) -> None:
        if self._map_shape is None:
            return
        el = self._model.add_line(self._map_shape)
        self._make_items(el)
        self._after_selection_change()

    def _add_roi(self) -> None:
        if self._map_shape is None:
            return
        el = self._model.add_roi(self._map_shape)
        self._make_items(el)
        self._after_selection_change()

    def _make_items(self, el) -> None:
        pg = self._pg
        pen = pg.mkPen(el.color, width=2.0)
        if el.kind == "line":
            roi = pg.LineSegmentROI(el.geom, pen=pen, movable=True)
            band = QtWidgets.QGraphicsPolygonItem()
            band.setPen(pg.mkPen("#FFEA00", width=1.0, style=QtCore.Qt.PenStyle.DashLine))
            band.setBrush(pg.mkBrush(255, 235, 59, 50))
            bar = pg.PlotDataItem(pen=pg.mkPen(el.color, width=3.0))
            head = QtWidgets.QGraphicsPolygonItem()
            head.setPen(pg.mkPen(el.color, width=1.0))
            head.setBrush(pg.mkBrush(el.color))
            deco = [band, bar, head]
        else:
            x0, x1, y0, y1 = el.geom[:4]
            roi = pg.ROI([x0, y0], [max(x1 - x0, 1.0), max(y1 - y0, 1.0)], pen=pen,
                         movable=True, rotatable=True, resizable=True)
            roi.addScaleHandle([1, 1], [0, 0])
            roi.addScaleHandle([0, 0], [1, 1])
            roi.addRotateHandle([1, 0.5], [0.5, 0.5])
            deco = []
            self._set_rect_state(roi, el.geom)
        roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        roi.sigClicked.connect(lambda *_a, eid=el.id: self._select(eid))
        roi.sigRegionChangeStarted.connect(lambda *_a, eid=el.id: self._select(eid))
        roi.sigRegionChanged.connect(lambda *_a, eid=el.id: self._on_item_moved(eid))
        self._vb.addItem(roi)
        for d in deco:
            d.setZValue(9)
            self._vb.addItem(d)
        label = pg.TextItem(el.id, color=el.color, anchor=(0, 1))
        label.setZValue(11)
        self._vb.addItem(label)
        self._items[el.id] = {"roi": roi, "deco": deco, "label": label}
        self._redraw_deco(el)

    def _remove_items(self, eid) -> None:
        it = self._items.pop(eid, None)
        if not it:
            return
        for g in [it["roi"], it["label"], *it["deco"]]:
            try:
                self._vb.removeItem(g)
            except Exception:
                pass

    @staticmethod
    def _set_rect_state(roi, geom) -> None:
        x0, x1, y0, y1 = geom[:4]
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        a = math.radians(E.roi_angle(geom))
        c, s = math.cos(a), math.sin(a)
        px = cx - (c * w / 2.0 - s * h / 2.0)
        py = cy - (s * w / 2.0 + c * h / 2.0)
        roi.setState({"pos": (px, py), "size": (w, h), "angle": E.roi_angle(geom)},
                     update=True)

    @staticmethod
    def _rect_geom(roi) -> list:
        st = roi.getState()
        (px, py), (w, h), ang = st["pos"], st["size"], float(st["angle"])
        a = math.radians(ang)
        c, s = math.cos(a), math.sin(a)
        cx = px + c * w / 2.0 - s * h / 2.0
        cy = py + s * w / 2.0 + c * h / 2.0
        ang = ((ang + 180.0) % 360.0) - 180.0
        return [cx - w / 2.0, cx + w / 2.0, cy - h / 2.0, cy + h / 2.0, ang]

    @staticmethod
    def _line_geom(roi) -> list:
        pts = [roi.mapToParent(h.pos()) for h in roi.getHandles()[:2]]
        return [[float(pts[0].x()), float(pts[0].y())], [float(pts[1].x()), float(pts[1].y())]]

    def _redraw_deco(self, el) -> None:
        it = self._items.get(el.id)
        if not it:
            return
        selected = el.id == self._model.selected
        it["roi"].setPen(self._pg.mkPen(el.color, width=3.5 if selected else 2.0))
        if el.kind == "line":
            (x0, y0), (x1, y1) = el.geom
            L = math.hypot(x1 - x0, y1 - y0)
            band, bar, head = it["deco"]
            H, W = self._map_shape or (100, 100)
            if L < 1e-6:
                band.setPolygon(QtGui.QPolygonF())
                head.setPolygon(QtGui.QPolygonF())
                bar.setData([], [])
            else:
                ux, uy = (x1 - x0) / L, (y1 - y0) / L
                px, py = -uy, ux
                half = (el.width - 1) / 2.0
                band.setPolygon(_poly([(x0 + px * half, y0 + py * half),
                                       (x1 + px * half, y1 + py * half),
                                       (x1 - px * half, y1 - py * half),
                                       (x0 - px * half, y0 - py * half)]))
                hb = max(3.0, 0.025 * min(H, W), half + 2.0)          # start bar
                bar.setData([x0 - px * hb, x0 + px * hb], [y0 - py * hb, y0 + py * hb])
                hl = max(4.0, 0.045 * min(H, W))                      # end arrowhead
                bx, by = x1 - ux * hl, y1 - uy * hl
                head.setPolygon(_poly([(x1, y1), (bx + px * hl * 0.5, by + py * hl * 0.5),
                                       (bx - px * hl * 0.5, by - py * hl * 0.5)]))
            it["label"].setPos(x0, y0)
        else:
            cs = E.roi_corners(el.geom)
            it["label"].setPos(float(cs[0][0]), float(cs[0][1]))

    def _on_item_moved(self, eid) -> None:
        if self._syncing:
            return
        el = self._model.get(eid)
        it = self._items.get(eid)
        if el is None or it is None:
            return
        el.geom = self._line_geom(it["roi"]) if el.kind == "line" else self._rect_geom(it["roi"])
        self._redraw_deco(el)
        if eid == self._model.selected:
            self._sync_panel()
        self._schedule()

    # ── selection / list ──────────────────────────────────────────────────────
    def _select(self, eid) -> None:
        if eid == self._model.selected:
            return
        self._model.select(eid)
        self._after_selection_change()

    def _after_selection_change(self) -> None:
        for el in self._model.elements:
            self._redraw_deco(el)
        self._refresh_list()
        self._sync_panel()
        self._schedule()

    def _refresh_list(self) -> None:
        self._elist.blockSignals(True)
        self._elist.clear()
        row = -1
        for i, el in enumerate(self._model.elements):
            it = QtWidgets.QListWidgetItem(f"{el.id}  ({'line' if el.kind == 'line' else 'ROI'})")
            it.setForeground(QtGui.QColor(el.color).darker(130))
            self._elist.addItem(it)
            if el.id == self._model.selected:
                row = i
        self._elist.setCurrentRow(row)
        self._elist.blockSignals(False)

    def _on_list_pick(self, row: int) -> None:
        if 0 <= row < len(self._model.elements):
            self._select(self._model.elements[row].id)

    def _delete_selected(self) -> None:
        eid = self._model.delete_selected()
        if eid is None:
            return
        self._remove_items(eid)
        self._after_selection_change()

    def _clear_all(self) -> None:
        if not self._model.elements:
            return
        ans = QtWidgets.QMessageBox.question(
            self, "Clear all", f"Delete all {len(self._model.elements)} element(s)?")
        if ans != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for el in list(self._model.elements):
            self._remove_items(el.id)
        self._model.clear()
        self._after_selection_change()

    # ── properties panel ──────────────────────────────────────────────────────
    def _sync_panel(self) -> None:
        el = self._model.selected_element()
        u, k = self._unit(), self._scale()
        for lab, name in self._line_labels + self._roi_labels:
            lab.setText(f"{name} ({u})")
        self._syncing = True
        try:
            if el is None:
                self._sel_title.setText("")
                self._forms.setCurrentIndex(2)
            elif el.kind == "line":
                p = MM.line_props(el.geom)
                self._sel_title.setText(f"{el.id} - line")
                for w, key in ((self._lx0, "x0"), (self._ly0, "y0"), (self._lx1, "x1"),
                               (self._ly1, "y1"), (self._llen, "length")):
                    w.setValue(p[key] * k)
                self._lw.setValue(el.width)
                self._forms.setCurrentIndex(0)
            else:
                p = MM.roi_props(el.geom)
                self._sel_title.setText(f"{el.id} - ROI")
                for w, key in ((self._rcx, "cx"), (self._rcy, "cy"),
                               (self._rlen, "length"), (self._rwid, "width")):
                    w.setValue(p[key] * k)
                self._rang.setValue(p["angle"])
                self._forms.setCurrentIndex(1)
        finally:
            self._syncing = False

    def _on_prop_edit(self, *_):
        if self._syncing:
            return
        el = self._model.selected_element()
        if el is None:
            return
        k = self._scale()
        if el.kind == "line":
            if self.sender() is self._llen:
                el.geom = MM.line_with_length(el.geom, self._llen.value() / k)
            else:
                el.geom = [[self._lx0.value() / k, self._ly0.value() / k],
                           [self._lx1.value() / k, self._ly1.value() / k]]
            el.width = self._model.set_width(el.id, self._lw.value())
            self._remove_items(el.id)                     # rebuild the segment handles
            self._make_items(el)
        else:
            el.geom = MM.roi_from_props(self._rcx.value() / k, self._rcy.value() / k,
                                        self._rlen.value() / k, self._rwid.value() / k,
                                        self._rang.value())
            self._syncing = True
            try:
                self._set_rect_state(self._items[el.id]["roi"], el.geom)
            finally:
                self._syncing = False
            self._redraw_deco(el)
        self._sync_panel()
        self._schedule()

    # ── drift ─────────────────────────────────────────────────────────────────
    def _poll_drift(self) -> None:
        sig = MM.drift_signature(self._scans)
        if sig == self._drift_sig:
            return
        self._drift_sig = sig
        if MM.any_drift(self._checked_scans() or self._scans):
            self._chk_drift.setChecked(True)
        self._schedule()

    def _load_drift(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load per-file drift CSV", "", "CSV (*.csv);;All (*)")
        if not p:
            return
        try:
            E.load_drift_csv(p, self._scans, log=self._log)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Drift CSV", f"Could not load:\n{exc}")
            return
        self._poll_drift()

    def _log(self, msg) -> None:
        con = getattr(self._host, "_console", None)
        if con is not None:
            con.log(msg)

    # ── plots ─────────────────────────────────────────────────────────────────
    def _mode(self) -> str:
        el = self._model.selected_element()
        if el is not None:
            return el.kind
        return "line" if self._model.lines() else "roi"

    def _update_plots(self) -> None:
        if self._drift_sig != MM.drift_signature(self._scans):      # drift re-read live
            self._poll_drift()
        tpl = self._tpl()
        scans = self._checked_scans()
        ch, lab = self._chan.currentData(), self._label.currentData() or "without_roi"
        use_drift = self._chk_drift.isChecked()
        clab = E._channel_label(tpl, ch) if tpl is not None else str(ch)
        self._prof.clear()
        self._means.clear()
        self._hist.clear()
        self._last_curves, self._last_rows = [], []
        if tpl is None or not scans:
            self._status.setText("Tick at least one file in the list.")
            return
        from engine import SIX_POINT_COLORS as COLS
        names = [s.name for s in scans]
        multi = len(scans) > 1
        if self._mode() == "line":
            self._plots.setCurrentIndex(0)
            curves = MM.line_curves(scans, tpl, self._model.lines(), ch, lab,
                                    use_drift=use_drift)
            k, u = self._scale(), self._unit()
            one_line = len(self._model.lines()) == 1
            for c in curves:
                el = self._model.get(c["line"])
                color = COLS[names.index(c["scan"]) % len(COLS)] if multi else el.color
                style = (QtCore.Qt.PenStyle.SolidLine
                         if one_line or c["line"] == self._model.selected
                         else QtCore.Qt.PenStyle.DotLine)
                self._prof.plot(c["dist_px"] * k, c["values"],
                                pen=self._pg.mkPen(color, width=2, style=style),
                                name=f"{c['line']} · {c['scan']}")
            self._prof.setLabel("left", clab)
            self._prof.setLabel("bottom", f"distance from start ({u})")
            self._last_curves = curves
            n = len(curves)
        else:
            self._plots.setCurrentIndex(1)
            rois = self._model.rois()
            rows = MM.roi_rows(scans, tpl, rois, ch, lab, use_drift=use_drift)
            ids = [r.id for r in rois]
            for si, sc in enumerate(scans):
                rs = [r for r in rows if r["scan"] == sc.name]
                if not rs:
                    continue
                xs = np.array([ids.index(r["roi"]) for r in rs], dtype=float)
                ys = np.array([r["mean"] for r in rs])
                sd = np.nan_to_num(np.array([r["sd"] for r in rs]))
                color = COLS[si % len(COLS)]
                self._means.plot(xs, ys, pen=self._pg.mkPen(color, width=2), symbol="o",
                                 symbolBrush=color, symbolSize=8, name=sc.name)
                self._means.addItem(self._pg.ErrorBarItem(
                    x=xs, y=ys, height=2 * sd, pen=self._pg.mkPen(color, width=1.5)))
            self._means.getAxis("bottom").setTicks([list(enumerate(ids))])
            self._means.setLabel("left", f"{clab}  mean ± SD")
            sel = self._model.selected if self._model.selected in ids else (ids[0] if ids else None)
            centers, counts = MM.histogram([r for r in rows if r["roi"] == sel])
            if len(centers) > 1:
                step = centers[1] - centers[0]
                edges = np.append(centers - step / 2.0, centers[-1] + step / 2.0)
                for si, sc in enumerate(scans):
                    if sc.name in counts:
                        color = COLS[si % len(COLS)]
                        self._hist.plot(edges, counts[sc.name], stepMode="center",
                                        pen=self._pg.mkPen(color, width=2), name=sc.name)
            self._hist.setLabel("bottom", f"{clab}  ({sel or '-'})")
            self._last_rows = rows
            n = len(rows)
        drift_note = "drift applied" if use_drift else "same pixel coordinates"
        self._status.setText(f"{clab} · {len(self._model.elements)} element(s) · {n} result(s) · "
                             f"{len(scans)} file(s) · {drift_note} · units {self._unit()}")

    # ── clipboard ─────────────────────────────────────────────────────────────
    def clipboard_text(self) -> str:
        tpl = self._tpl()
        ch = self._chan.currentData()
        vu = _value_unit(E._channel_label(tpl, ch) if tpl is not None else str(ch))
        if self._mode() == "line":
            return MM.profile_text(self._last_curves, self._unit(), vu, scale=self._scale())
        rows = self._last_rows
        sel = self._model.selected
        centers, counts = MM.histogram([r for r in rows if r["roi"] == sel])
        txt = MM.roi_text(rows, vu)
        if len(centers):
            txt += "\n\n" + MM.histogram_text(centers, counts, vu)
        return txt

    def _copy(self) -> None:
        txt = self.clipboard_text()
        if txt:
            QtWidgets.QApplication.clipboard().setText(txt)
            self._status.setText(self._status.text().split("  [copied")[0]
                                 + f"  [copied {txt.count(chr(10)) + 1} rows]")

    # ── Report ────────────────────────────────────────────────────────────────
    def _send_to_report(self) -> None:
        tpl = self._tpl()
        scans = self._checked_scans()
        if tpl is None or not self._model.elements:
            QtWidgets.QMessageBox.information(self, "Send to Report", "Add a line or ROI first.")
            return
        if tpl not in scans:
            QtWidgets.QMessageBox.warning(self, "Send to Report",
                                          "Tick the template file too: the geometry comes from it.")
            return
        use_drift = self._chk_drift.isChecked()
        ch, lab = self._chan.currentData(), self._label.currentData() or "without_roi"
        removed = self._loaded_ids - {e.id for e in self._model.elements}
        for sc in scans:                                  # elements deleted in the tool
            for eid in removed:
                (getattr(sc, "lines", None) or {}).pop(eid, None)
                (getattr(sc, "line_widths", None) or {}).pop(eid, None)
                (getattr(sc, "area_rois", None) or {}).pop(eid, None)
        for el in self._model.lines():
            E.place_line_from_spec(scans, el.id, {"type": "seg", "p0": el.geom[0],
                                                  "p1": el.geom[1]},
                                   template_scan=tpl, use_drift=use_drift, log=self._log)
            for sc in scans:
                sc.line_widths = dict(getattr(sc, "line_widths", None) or {})
                sc.line_widths[el.id] = el.width
            E.register_live_line_report_figures(scans, [el.id], channel=ch, label=lab,
                                                width=el.width, log=self._log)
        rois = self._model.rois()
        for el in rois:
            E.place_roi_from_spec(scans, el.id, el.geom, template_scan=tpl,
                                  use_drift=use_drift, log=self._log)
        if rois:
            E.register_live_roi_report_figures(scans, [e.id for e in rois], channel=ch,
                                               label=lab, log=self._log)
        self._loaded_ids = {e.id for e in self._model.elements}
        try:
            self._host._update_active_views()
            self._host._params.report_refresh()
            report = self._host._params.report
            keys = [k for k in E.list_figure_keys(tpl) if k.startswith("report_")]
            if report is not None and keys:
                report.select_figure(self._host._scans.index(tpl), keys[0])
                self._host._params._tabs.setCurrentWidget(report)
        except Exception:
            pass
        ids = ", ".join(e.id for e in self._model.elements)
        self._log(f"Line/ROI tool → Report: {ids} on {len(scans)} file(s), {ch}/{lab}"
                  + (f"; removed {', '.join(sorted(removed))}" if removed else ""))
        QtWidgets.QMessageBox.information(self, "Send to Report",
                                          f"Sent {ids} to {len(scans)} file(s).")
