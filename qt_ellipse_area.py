"""qt_ellipse_area -- the "Ellipse from area" window.

Fit the detector ellipse on diffraction with isotropic rings instead of on the Bragg vector
map of a single crystal (where an elliptical detector cannot be told apart from strain):
1. draw one or more areas over amorphous / polycrystalline material on the ADF (rotatable
   rectangle, ellipse, free polygon; any area can be marked "exclude"), or choose the
   whole-scan DP mean / DP max;
2. Build image: mean of the raw patterns in the areas, each re-centred on the beam first;
3. pick the ring (suggested from the radial profile) and Fit (amorphous halo or sharp ring);
   the radial profile before / after the correction shows whether the ring sharpens;
4. Apply to the ticked files (same camera length / optics), and Verify on another file.
Numbers: ellipse_area.py and engine (ellipse_source_image, fit_ellipse_from_source,
apply_area_ellipse, verify_area_ellipse). Coordinates: x = scan column, y = scan row on the
ADF; qx = row, qy = column on the pattern.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

import engine as E
import ellipse_area as EA

SOURCES = (("Area (raw patterns)", "area"), ("DP mean (whole scan)", "dp_mean"),
           ("DP max (whole scan)", "dp_max"))
METHODS = (("Amorphous halo", "amorphous"), ("Sharp ring", "ring"))
COLORS = ["#00E5FF", "#FF6D00", "#76FF03", "#E040FB", "#FFEA00", "#FF4081"]


class EllipseAreaDialog(QtWidgets.QDialog):
    def __init__(self, host) -> None:
        super().__init__(host)
        try:
            from qt_main import _enable_minmax
            _enable_minmax(self)
        except Exception:
            pass
        import pyqtgraph as pg
        self._pg = pg
        self._host = host
        self._scans = list(getattr(host, "_scans", None) or [])
        self._sc = host.active_scan() if hasattr(host, "active_scan") else (
            self._scans[0] if self._scans else None)
        self._areas: list[dict] = []          # {"kind", "roi", "exclude"}
        self._src: dict | None = None
        self._fit: dict | None = None
        self._adf_shape: tuple | None = None
        self.setWindowTitle(f"Ellipse from area -- {getattr(self._sc, 'name', '')}")
        self.resize(1400, 820)
        self._build(pg)
        self._load_adf()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self, pg) -> None:
        pg.setConfigOptions(imageAxisOrder="row-major")
        lay = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("<b>1. Areas</b> on the ADF (amorphous / polycrystalline)"))
        self._glw_adf = pg.GraphicsLayoutWidget()
        self._vb_adf = self._glw_adf.addViewBox()
        self._vb_adf.setAspectLocked(True)
        self._vb_adf.invertY(True)
        self._img_adf = pg.ImageItem()
        self._vb_adf.addItem(self._img_adf)
        left.addWidget(self._glw_adf, 1)
        row = QtWidgets.QHBoxLayout()
        for txt, kind in (("+ Rectangle", "rect"), ("+ Ellipse", "ellipse"), ("+ Polygon", "polygon")):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(lambda _=False, k=kind: self.add_area(k))
            row.addWidget(b)
        left.addLayout(row)
        self._list = QtWidgets.QListWidget()
        self._list.setToolTip("Ticked = EXCLUDE this area (e.g. a crystal inside the amorphous zone)")
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.setMaximumHeight(120)
        left.addWidget(self._list)
        b_del = QtWidgets.QPushButton("Delete selected area")
        b_del.clicked.connect(lambda: self.delete_area(self._list.currentRow()))
        left.addWidget(b_del)
        lay.addLayout(left, 3)

        mid = QtWidgets.QVBoxLayout()
        mid.addWidget(QtWidgets.QLabel("<b>2. Image</b> (log) with the ring and the fitted ellipse"))
        self._glw_dp = pg.GraphicsLayoutWidget()
        self._vb_dp = self._glw_dp.addViewBox()
        self._vb_dp.setAspectLocked(True)
        self._vb_dp.invertY(True)
        self._img_dp = pg.ImageItem()
        self._vb_dp.addItem(self._img_dp)
        self._c0 = pg.PlotDataItem(pen=pg.mkPen("#FFEB3B", width=1.2))
        self._c1 = pg.PlotDataItem(pen=pg.mkPen("#FFEB3B", width=1.2))
        self._ell = pg.PlotDataItem(pen=pg.mkPen("#00E5FF", width=2.0))
        for it in (self._c0, self._c1, self._ell):
            it.setZValue(10)
            self._vb_dp.addItem(it)
        mid.addWidget(self._glw_dp, 3)
        self._prof = pg.PlotWidget()
        self._prof.addLegend(offset=(-10, 10))
        self._prof.setLabel("bottom", "radius (px)")
        self._prof.setLabel("left", "mean intensity")
        self._cur_before = self._prof.plot(pen=pg.mkPen("#9E9E9E", width=1.5), name="circular")
        self._cur_after = self._prof.plot(pen=pg.mkPen("#00ACC1", width=2.0), name="ellipse-corrected")
        mid.addWidget(self._prof, 2)
        lay.addLayout(mid, 4)

        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(330)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel(f"<b>File:</b> {getattr(self._sc, 'name', '-')}"))
        self._cb_source = QtWidgets.QComboBox()
        for txt, key in SOURCES:
            self._cb_source.addItem(txt, key)
        self._cb_source.currentIndexChanged.connect(lambda *_: self._sync_source())
        v.addWidget(QtWidgets.QLabel("Source"))
        v.addWidget(self._cb_source)
        self._chk_recenter = QtWidgets.QCheckBox("Re-centre every pattern on the beam")
        self._chk_recenter.setChecked(True)
        self._chk_recenter.setToolTip("Removes the descan blur of the ring (Origin calibration "
                                      "if done, else the direct-beam centre of mass)")
        v.addWidget(self._chk_recenter)
        b_build = QtWidgets.QPushButton("Build image")
        b_build.clicked.connect(self.build_image)
        v.addWidget(b_build)
        v.addWidget(QtWidgets.QLabel("<b>3. Fit</b>"))
        self._cb_method = QtWidgets.QComboBox()
        for txt, key in METHODS:
            self._cb_method.addItem(txt, key)
        v.addWidget(self._cb_method)
        self._sp_r0 = self._spin(v, "ring r0 (px)", 10.0)
        self._sp_r1 = self._spin(v, "ring r1 (px)", 30.0)
        for s in (self._sp_r0, self._sp_r1):
            s.valueChanged.connect(lambda *_: self._draw_ring())
        b_fit = QtWidgets.QPushButton("Fit ellipse")
        b_fit.clicked.connect(self.fit)
        v.addWidget(b_fit)
        v.addWidget(QtWidgets.QLabel("<b>4. Apply</b> to the ticked files (same CL / optics)"))
        self._targets = QtWidgets.QListWidget()
        self._targets.setMaximumHeight(130)
        for sc in self._scans:
            it = QtWidgets.QListWidgetItem(sc.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if sc is self._sc
                             else QtCore.Qt.CheckState.Unchecked)
            self._targets.addItem(it)
        v.addWidget(self._targets)
        b_apply = QtWidgets.QPushButton("Apply to ticked files")
        b_apply.clicked.connect(self.apply)
        v.addWidget(b_apply)
        v.addWidget(QtWidgets.QLabel("Verify on another file (same areas)"))
        self._cb_verify = QtWidgets.QComboBox()
        for sc in self._scans:
            self._cb_verify.addItem(sc.name)
        v.addWidget(self._cb_verify)
        b_ver = QtWidgets.QPushButton("Verify")
        b_ver.clicked.connect(lambda: self.verify(self._cb_verify.currentText()))
        v.addWidget(b_ver)
        v.addStretch(1)
        self._status = QtWidgets.QLabel("Draw areas over amorphous / polycrystalline material, "
                                        "or choose DP mean / max, then Build image.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        v.addWidget(bb)
        lay.addWidget(panel)

    @staticmethod
    def _spin(layout, label, val):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(0.0, 2000.0)
        s.setDecimals(1)
        s.setSingleStep(0.5)
        s.setValue(val)
        row.addWidget(s, 1)
        layout.addLayout(row)
        return s

    def _log(self, msg) -> None:
        con = getattr(self._host, "_console", None)
        if con is not None:
            con.log(msg)

    # ── ADF and areas ─────────────────────────────────────────────────────────
    def _load_adf(self) -> None:
        adf = E.cached_adf(self._sc) if self._sc is not None else None
        if adf is None and self._sc is not None:
            try:
                adf = E.load_adf(self._sc)
            except Exception:
                adf = None
        if adf is None:
            self._status.setText("No ADF for this file: areas cannot be drawn. Use DP mean / max, "
                                 "or create the virtual images first.")
            return
        a = np.asarray(adf, dtype=float)
        self._adf_shape = a.shape
        self._img_adf.setImage(a)
        fin = a[np.isfinite(a)]
        if fin.size:
            self._img_adf.setLevels([float(np.percentile(fin, 1)), float(np.percentile(fin, 99))])

    def _default_geom(self, kind):
        H, W = self._adf_shape or (32, 32)
        if kind == "rect":
            return [W * 0.1, W * 0.4, H * 0.1, H * 0.4, 0.0]
        if kind == "ellipse":
            return {"center": [W * 0.5, H * 0.5], "radii": [W * 0.12, H * 0.12], "angle": 0.0}
        return [[W * 0.2, H * 0.6], [W * 0.45, H * 0.6], [W * 0.32, H * 0.85]]

    def add_area(self, kind: str, geom=None) -> int:
        from qt_measure import MeasureDialog
        pg = self._pg
        kind = str(kind).lower()
        geom = self._default_geom(kind) if geom is None else geom
        pen = pg.mkPen(COLORS[len(self._areas) % len(COLORS)], width=2)
        if kind == "rect":
            roi = pg.ROI([0, 0], [1, 1], pen=pen)
            roi.addScaleHandle([1, 1], [0, 0])
            roi.addScaleHandle([0, 0], [1, 1])
            roi.addRotateHandle([1, 0.5], [0.5, 0.5])
            MeasureDialog._set_rect_state(roi, list(geom))
        elif kind == "ellipse":
            (cx, cy), (rx, ry) = geom["center"], geom["radii"]
            roi = pg.EllipseROI([0, 0], [1, 1], pen=pen)
            MeasureDialog._set_rect_state(roi, [cx - rx, cx + rx, cy - ry, cy + ry,
                                                float(geom.get("angle", 0.0))])
        elif kind == "polygon":
            roi = pg.PolyLineROI([list(map(float, p)) for p in geom], closed=True, pen=pen)
        else:
            raise ValueError(f"unknown area kind {kind!r}")
        self._vb_adf.addItem(roi)
        self._areas.append({"kind": kind, "roi": roi, "exclude": False})
        it = QtWidgets.QListWidgetItem(f"A{len(self._areas)}  {kind}")
        it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self._list.blockSignals(True)
        self._list.addItem(it)
        self._list.blockSignals(False)
        return len(self._areas) - 1

    def areas(self) -> list:
        from qt_measure import MeasureDialog
        out = []
        for a in self._areas:
            roi = a["roi"]
            if a["kind"] == "rect":
                g = MeasureDialog._rect_geom(roi)
                d = {"kind": "rect", "bounds": [float(v) for v in g[:4]], "angle": float(g[4])}
            elif a["kind"] == "ellipse":
                x0, x1, y0, y1, ang = MeasureDialog._rect_geom(roi)
                d = {"kind": "ellipse", "center": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
                     "radii": [(x1 - x0) / 2.0, (y1 - y0) / 2.0], "angle": float(ang)}
            else:
                pts = [roi.mapToParent(h.pos()) for h in roi.getHandles()]
                d = {"kind": "polygon", "points": [[float(p.x()), float(p.y())] for p in pts]}
            d["exclude"] = bool(a["exclude"])
            out.append(d)
        return out

    def set_exclude(self, i: int, flag: bool) -> None:
        self._areas[i]["exclude"] = bool(flag)
        it = self._list.item(i)
        self._list.blockSignals(True)
        it.setCheckState(QtCore.Qt.CheckState.Checked if flag else QtCore.Qt.CheckState.Unchecked)
        self._list.blockSignals(False)
        pen = self._areas[i]["roi"].pen
        style = QtCore.Qt.PenStyle.DashLine if flag else QtCore.Qt.PenStyle.SolidLine
        try:
            pen.setStyle(style)
            self._areas[i]["roi"].setPen(pen)
        except Exception:
            pass

    def _on_item_changed(self, item) -> None:
        i = self._list.row(item)
        if 0 <= i < len(self._areas):
            self.set_exclude(i, item.checkState() == QtCore.Qt.CheckState.Checked)

    def delete_area(self, i: int) -> None:
        if not (0 <= i < len(self._areas)):
            return
        self._vb_adf.removeItem(self._areas[i]["roi"])
        self._areas.pop(i)
        self._list.blockSignals(True)
        self._list.takeItem(i)
        self._list.blockSignals(False)

    # ── source / image ────────────────────────────────────────────────────────
    def _source(self) -> str:
        return str(self._cb_source.currentData())

    def set_source(self, key: str) -> None:
        for i in range(self._cb_source.count()):
            if self._cb_source.itemData(i) == key:
                self._cb_source.setCurrentIndex(i)
        self._sync_source()

    def _sync_source(self) -> None:
        self._chk_recenter.setEnabled(self._source() == "area")

    def build_image(self) -> None:
        if self._sc is None:
            return
        sc, src = self._sc, self._source()
        areas = self.areas() if src == "area" else None
        rec = self._chk_recenter.isChecked() if src == "area" else False
        if src == "area" and not areas:
            self._status.setText("Draw at least one area first.")
            return
        self._status.setText("Building the image...")

        def work():
            return E.ellipse_source_image(sc, src, areas=areas, recenter=rec, log=self._log)

        self._host._run_async(work, label=f"Ellipse source ({sc.name})", on_done=self._on_built)

    def _on_built(self, res) -> None:
        if isinstance(res, Exception):
            self._status.setText(f"Image not built: {res}")
            return
        self._src, self._fit = res, None
        img = np.asarray(res["image"], dtype=float)
        disp = np.log1p(np.clip(img, 0, None))
        self._img_dp.setImage(disp)
        self._img_dp.setLevels([float(np.percentile(disp, 1)), float(np.percentile(disp, 99.8))])
        c = self._center()
        r, prof = EA.radial_profile(img, c)
        k = 2
        while k < len(prof) - 1 and prof[k + 1] < prof[k]:
            k += 1                                            # end of the direct-beam fall
        try:
            r0, r1 = EA.suggest_ring(r, prof, r_min=float(r[k]) + 1.0, half_width=6.0)
            self._sp_r0.setValue(r0)
            self._sp_r1.setValue(r1)
        except ValueError:
            pass
        self._cur_before.setData(r, prof)
        self._cur_after.setData([], [])
        self._ell.setData([], [])
        self._draw_ring()
        n = res.get("n_patterns")
        self._status.setText(
            f"Image ready: {res['source']}" + (f", {n} patterns" if n else "")
            + (f", re-centred ({res.get('origin_source')})" if res.get("recentered") else ", not re-centred")
            + f". Ring suggested at {self._sp_r0.value():.1f}-{self._sp_r1.value():.1f} px: check it, then Fit.")

    def _center(self):
        if self._src is None:
            return (0.0, 0.0)
        c = self._src.get("center")
        if c is None:
            s = np.asarray(self._src["image"]).shape
            c = (s[0] / 2.0, s[1] / 2.0)
        return float(c[0]), float(c[1])

    def ring_range(self) -> tuple[float, float]:
        return float(self._sp_r0.value()), float(self._sp_r1.value())

    def _draw_ring(self) -> None:
        if self._src is None:
            return
        cx, cy = self._center()                               # (qx = row, qy = col)
        t = np.linspace(0, 2 * np.pi, 181)
        for item, r in ((self._c0, self._sp_r0.value()), (self._c1, self._sp_r1.value())):
            item.setData(cy + r * np.cos(t), cx + r * np.sin(t))   # view x = column

    # ── fit / apply / verify ──────────────────────────────────────────────────
    def fit(self) -> None:
        if self._src is None:
            self._status.setText("Build the image first.")
            return
        sc, src, rr = self._sc, self._src, self.ring_range()
        method = str(self._cb_method.currentData())

        def work():
            return E.fit_ellipse_from_source(sc, src, rr, method, log=self._log)

        self._host._run_async(work, label=f"Ellipse fit ({sc.name})", on_done=self._on_fit)

    def _on_fit(self, res) -> None:
        if isinstance(res, Exception):
            self._status.setText(f"Fit failed: {res}")
            return
        self._fit = res
        qx0, qy0, a, b, th = res["p_ellipse"]
        t = np.linspace(0, 2 * np.pi, 361)
        ex = a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th)     # along qx
        ey = a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)     # along qy
        self._ell.setData(qy0 + ey, qx0 + ex)
        img = np.asarray(self._src["image"], dtype=float)
        c = self._center()
        r0, prof0 = EA.radial_profile(img, c)
        r1, prof1 = EA.radial_profile(img, c, p_ellipse=res["p_ellipse"])
        self._cur_after.setData(r1, prof1)
        w0 = EA.peak_fwhm(r0, prof0, res["r_range"])
        w1 = EA.peak_fwhm(r1, prof1, res["r_range"])
        q = res["quality"]
        self._status.setText(
            f"a/b = {res['a_over_b']:.4f}, major axis at {res['theta_deg']:.2f} deg "
            f"({res['method']}). Ring FWHM {w0:.2f} -> {w1:.2f} px after correction. "
            + ("Quality OK: tick the files and Apply." if q["ok"]
               else "REJECTED: " + "; ".join(q["reasons"])))

    def set_targets(self, names) -> None:
        names = set(names)
        for i in range(self._targets.count()):
            it = self._targets.item(i)
            it.setCheckState(QtCore.Qt.CheckState.Checked if it.text() in names
                             else QtCore.Qt.CheckState.Unchecked)

    def _ticked(self) -> list:
        names = {self._targets.item(i).text() for i in range(self._targets.count())
                 if self._targets.item(i).checkState() == QtCore.Qt.CheckState.Checked}
        return [sc for sc in self._scans if sc.name in names]

    def apply(self) -> None:
        if self._fit is None:
            self._status.setText("Fit the ellipse first.")
            return
        if not self._fit["quality"]["ok"]:
            self._status.setText("The fit was rejected by the quality control: not applied.")
            return
        targets = self._ticked()
        if not targets:
            self._status.setText("Tick at least one file.")
            return
        areas = self.areas() if self._src.get("source") == "area" else None
        now = E.apply_area_ellipse(targets, self._fit, self._src, areas=areas,
                                   source_scan=self._sc, log=self._log)
        try:
            self._host._params.reload()
        except Exception:
            pass
        try:
            self._host._update_active_views()
        except Exception:
            pass
        self._status.setText(
            f"Ellipse a/b = {self._fit['a_over_b']:.4f} stored on {len(targets)} file(s) "
            f"({', '.join(sc.name for sc in targets)}); applied now on {len(now)}, the others "
            f"when they are calibrated / computed.")

    def verify(self, name: str) -> None:
        if self._fit is None:
            self._status.setText("Fit the ellipse first.")
            return
        sc = next((s for s in self._scans if s.name == name), None)
        areas = self.areas()
        if sc is None or not areas:
            self._status.setText("Pick a file and draw the areas to check on it.")
            return
        fit = self._fit
        method = fit.get("method", "amorphous")

        def work():
            return E.verify_area_ellipse(sc, areas, fit["p_ellipse"], fit["r_range"], method,
                                         log=self._log)

        def done(res):
            if isinstance(res, Exception):
                self._status.setText(f"Verify failed on {name}: {res}")
                return
            self._status.setText(
                f"{name}: own a/b = {res['measured_a_over_b']:.4f}; after the applied ellipse "
                f"residual a/b = {res['residual_a_over_b']:.4f} -> "
                + ("OK, the ellipse holds for this file." if res["ok"]
                   else f"DOES NOT HOLD (tolerance {res['tol']}): fit this file on its own."))

        self._host._run_async(work, label=f"Ellipse verify ({name})", on_done=done)
