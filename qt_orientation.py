"""qt_orientation — OrientationPeaksDialog: py4DSTEM orientation → peaks (Plugin).

Modeless dialog opened from **Plugin → Orient. peaks…**. Runs
``engine.run_orientation_peaks`` without touching the strain pipeline. Optional
Compare vs Index BVM and Send writes the same ``index_origin/g1/g2`` contract.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

import engine as E


def _enable_minmax(dlg) -> None:
    try:
        dlg.setWindowFlag(QtCore.Qt.WindowType.WindowMinimizeButtonHint, True)
        dlg.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint, True)
    except Exception:
        pass


def _parse_ivec3(text: str, default: list[int]) -> list[int]:
    parts = [
        p for p in (text or "").replace(";", ",").replace("[", "").replace("]", "").split(",")
        if p.strip() != ""
    ]
    if len(parts) != 3:
        return list(default)
    return [int(round(float(p))) for p in parts]


class OrientationPeaksDialog(QtWidgets.QDialog):
    """py4DSTEM Path A/B orientation → peaks; side-by-side with Index BVM."""

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
        self._result = None
        self._compare_info: dict | None = None
        self.setWindowTitle(f"Orient. peaks — {self._sc.name if self._sc else ''}")
        self.resize(1280, 860)

        lay = QtWidgets.QHBoxLayout(self)
        self._fig_host = QtWidgets.QVBoxLayout()
        fw = QtWidgets.QWidget()
        fw.setLayout(self._fig_host)
        lay.addWidget(fw, 1)

        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(380)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel(
            "<b>Orient. peaks</b> — Plugin: py4DSTEM Crystal "
            "(Path A: generate · Path B: ACOM)."
        ))

        p = self._sc.params if self._sc else E.CalibrationParams()
        self._lbl_crystal = QtWidgets.QLabel("")
        self._lbl_crystal.setWordWrap(True)
        self._lbl_cif_warn = QtWidgets.QLabel("")
        self._lbl_cif_warn.setWordWrap(True)
        self._lbl_cif_warn.setStyleSheet("color:#E65100; font-size:11px;")
        v.addWidget(self._lbl_crystal)
        v.addWidget(self._lbl_cif_warn)
        btn_cif = QtWidgets.QPushButton("Load CIF…")
        btn_cif.clicked.connect(self._load_cif)
        v.addWidget(btn_cif)
        self._refresh_crystal_label()

        row_mode = QtWidgets.QHBoxLayout()
        row_mode.addWidget(QtWidgets.QLabel("Method"))
        self._cmb_mode = QtWidgets.QComboBox()
        self._cmb_mode.addItem("Known — generate pattern", "known_generate")
        self._cmb_mode.addItem("ACOM — match_single_pattern", "acom_match")
        self._cmb_mode.setToolTip(
            "Known: zone + proj_x → generate_diffraction_pattern → match BVM.\n"
            "ACOM: orientation_plan + match_single_pattern → regenerate → match."
        )
        self._cmb_mode.currentIndexChanged.connect(self._on_mode)
        row_mode.addWidget(self._cmb_mode, 1)
        v.addLayout(row_mode)

        self._ed_zone, self._lbl_zone = self._line_row(
            v, "Zone axis [uvw]",
            ",".join(str(int(x)) for x in (p.zone_axis or [1, 1, 0])),
            return_label=True,
        )
        self._ed_proj, self._lbl_proj = self._line_row(
            v, "proj_x lattice",
            ",".join(str(int(x)) for x in (p.real_axis_h or [0, 0, -1])),
            return_label=True,
        )
        self._sp_kmax = self._dspin_row(v, "k_max (Å⁻¹)", 0.2, 5.0, 1.2, 0.1, 2)
        tol0 = float(p.indexing_tol_px) if p.indexing_tol_px is not None else float(p.max_peak_spacing)
        self._sp_tol = self._dspin_row(v, "Tolerance (px)", 0.1, 50.0, tol0, 0.1, 2)
        self._sp_za = self._dspin_row(v, "ACOM za step (°)", 1.0, 15.0, 4.0, 1.0, 1)
        self._sp_ip = self._dspin_row(v, "ACOM in-plane (°)", 1.0, 15.0, 4.0, 1.0, 1)

        row_up = QtWidgets.QHBoxLayout()
        row_up.addWidget(QtWidgets.QLabel("Peak sampling"))
        self._cmb_upsample = QtWidgets.QComboBox()
        for f in (1, 2, 4):
            self._cmb_upsample.addItem(f"{f}×", f)
        self._cmb_upsample.setCurrentIndex(0)
        self._cmb_upsample.setToolTip(
            "Spatial upsampling of the BVM before peak detection (sub-pixel)."
        )
        row_up.addWidget(self._cmb_upsample, 1)
        v.addLayout(row_up)

        self._on_mode()

        for txt, fn, tip in (
            ("Run", self._run, "Run py4DSTEM Path A or B on the calibrated BVM."),
            ("Compare vs Index BVM", self._compare,
             "Overlay / metrics vs our Index BVM (runs Index if needed). Does not Send."),
            ("Send to Fast4D", self._send,
             "Write index_origin/g1/g2 + manual_enabled, and suggested "
             "QR_rotation + coordinate_rotation from CIF↔measured fit."),
            ("Export CSV/PNG", self._export, "Save match table + overlay."),
        ):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            b.setToolTip(tip)
            if txt == "Send to Fast4D":
                b.setStyleSheet(
                    "QPushButton{background:#E8F5E9;border:1px solid #2E7D32;"
                    "border-radius:5px;padding:6px;font-weight:600;}"
                )
            v.addWidget(b)

        self._lbl_propose = QtWidgets.QLabel("Proposed: —")
        self._lbl_propose.setWordWrap(True)
        self._lbl_propose.setStyleSheet("color:#0D47A1; font-size:11px;")
        v.addWidget(self._lbl_propose)

        self._lbl_compare = QtWidgets.QLabel("")
        self._lbl_compare.setWordWrap(True)
        self._lbl_compare.setStyleSheet("color:#6A1B9A; font-size:11px;")
        v.addWidget(self._lbl_compare)

        self._table = QtWidgets.QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["#", "hkl", "qx_px", "qy_px", "res_px", "I", "theo|g|"]
        )
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMaximumHeight(260)
        v.addWidget(self._table, 1)

        self._status = QtWidgets.QLabel("Ready — choose method and Run.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        v.addWidget(bb)
        lay.addWidget(panel)

    def _line_row(self, layout, label, text, *, return_label: bool = False):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        lbl.setMinimumWidth(130)
        ed = QtWidgets.QLineEdit(text)
        row.addWidget(lbl)
        row.addWidget(ed, 1)
        layout.addLayout(row)
        if return_label:
            return ed, lbl
        return ed

    def _dspin_row(self, layout, label, lo, hi, val, step, dec):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        lbl.setMinimumWidth(130)
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(int(dec))
        s.setRange(float(lo), float(hi))
        s.setSingleStep(float(step))
        s.setValue(float(val))
        row.addWidget(lbl)
        row.addWidget(s, 1)
        layout.addLayout(row)
        return s

    def _mode(self) -> str:
        return str(self._cmb_mode.currentData() or "known_generate")

    def _on_mode(self, *_args) -> None:
        known = self._mode() == "known_generate"
        for w in (self._ed_zone, self._lbl_zone, self._ed_proj, self._lbl_proj):
            w.setEnabled(known)
        for w in (self._sp_za, self._sp_ip):
            w.setEnabled(not known)

    def _refresh_crystal_label(self) -> None:
        p = self._sc.params if self._sc else E.CalibrationParams()
        warn = ""
        try:
            if p.cal_crystal == "CIF" and p.cif_path:
                info = E.load_crystal_from_cif(p.cif_path)
                crystal = info.cal
                src = Path(info.path).name
                if info.warning:
                    warn = info.warning
                text = (
                    f"Crystal: <b>{crystal.name}</b> (CIF)  a = {crystal.a_lat:.4f} Å"
                    f"<br/><span style='font-size:10px;color:#555;'>{src}</span>"
                )
            else:
                crystal = p.cal_crystal_obj()
                text = f"Crystal: <b>{crystal.name}</b>  a = {crystal.a_lat:.4f} Å"
        except Exception as exc:
            crystal = E.CAL_CRYSTALS[E.DEFAULT_CAL_CRYSTAL]
            text = f"Crystal: <b>{crystal.name}</b>  a = {crystal.a_lat:.4f} Å"
            warn = str(exc)
        self._lbl_crystal.setText(text)
        self._lbl_cif_warn.setText(warn)
        self._lbl_cif_warn.setVisible(bool(warn))

    def _load_cif(self) -> None:
        if self._sc is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load crystal CIF",
            str(Path(self._sc.params.cif_path).parent) if self._sc.params.cif_path else "",
            "CIF files (*.cif);;All files (*.*)",
        )
        if not path:
            return
        try:
            info = E.load_crystal_from_cif(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "CIF load failed", str(exc))
            return
        self._sc.params.cal_crystal = "CIF"
        self._sc.params.cif_path = info.path
        try:
            self._host._params.reload()
        except Exception:
            pass
        self._refresh_crystal_label()
        self._status.setText(f"Loaded CIF: {Path(info.path).name}")

    def _collect_params(self) -> None:
        p = self._sc.params
        p.zone_axis = _parse_ivec3(self._ed_zone.text(), [1, 1, 0])
        p.real_axis_h = _parse_ivec3(self._ed_proj.text(), [0, 0, -1])
        p.indexing_tol_px = float(self._sp_tol.value())

    def _show_fig(self, fig, *, owned: bool = True):
        if fig is None:
            return
        prev = getattr(self, "_preview_fig", None)
        if prev is not None and prev is not fig and getattr(self, "_preview_owned", True):
            E.close_figure(prev)
        if self._canvas is not None:
            self._fig_host.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None
        if self._tb is not None:
            self._fig_host.removeWidget(self._tb)
            self._tb.setParent(None)
            self._tb.deleteLater()
            self._tb = None
        self._preview_fig = fig
        self._preview_owned = bool(owned)
        self._canvas = self._FC(fig)
        self._tb = self._safe_tb(self._canvas, self)
        self._fig_host.addWidget(self._tb)
        self._fig_host.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def _fill_table(self, result) -> None:
        rows = sorted(result.matched, key=lambda m: m.intensity, reverse=True)
        self._table.setRowCount(len(rows))
        for r, m in enumerate(rows):
            g = (m.theo_qx_A ** 2 + m.theo_qy_A ** 2) ** 0.5
            vals = [
                str(m.measured_index),
                f"({m.h} {m.k} {m.l})",
                f"{m.qx_px:.2f}",
                f"{m.qy_px:.2f}",
                f"{m.residual_px:.3f}",
                f"{m.intensity:.1f}",
                f"{g:.4f}",
            ]
            for c, text in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if m.measured_index in (result.index_g1, result.index_g2, result.index_origin):
                    it.setBackground(QtGui.QColor("#E3F2FD"))
                self._table.setItem(r, c, it)
        self._table.resizeColumnsToContents()
        corr = f"  corr={result.corr_score:.3f}" if result.corr_score is not None else ""
        self._lbl_propose.setText(
            f"Proposed: origin={result.index_origin}  g1={result.index_g1}  "
            f"g2={result.index_g2}  matched={result.n_matched}/{result.n_theoretical}  "
            f"rms={result.rms_px:.3f} px{corr}  [{result.mode}]"
        )
        qr = getattr(result, "suggested_qr_rotation_deg", None)
        if qr is not None:
            self._lbl_propose.setText(
                self._lbl_propose.text()
                + f"\nSuggested QR_rotation = coordinate_rotation ≈ {float(qr):.2f}° (Send writes both)"
            )

    def _run(self) -> None:
        if self._sc is None or self._host._busy:
            return
        self._collect_params()
        sc = self._sc
        mode = self._mode()
        k_max = float(self._sp_kmax.value())
        tol = float(self._sp_tol.value())
        za = float(self._sp_za.value())
        ip = float(self._sp_ip.value())

        def work():
            up = int(self._cmb_upsample.currentData() or 1)
            return E.run_orientation_peaks(
                sc,
                mode=mode,
                k_max=k_max,
                tol_px=tol,
                angle_step_zone_axis=za,
                angle_step_in_plane=ip,
                image_upsample=up,
                make_figure=False,
                log=self._host._console.log,
            )

        def on_done(r):
            if isinstance(r, Exception):
                self._status.setText(f"Error: {r}")
                return
            self._result = r
            self._compare_info = None
            self._lbl_compare.setText("")
            try:
                import orientation_peaks as op
                fig = op.make_orientation_peaks_figure(
                    r,
                    title=f"{sc.name} — Orient. peaks [{r.mode}]",
                    indexing_result=sc.indexing_result,
                )
                self._show_fig(fig, owned=True)
                E.register_figure(sc, "orientation_peaks", fig, force=True)
                self._preview_owned = False
            except Exception as exc:
                self._status.setText(f"Done, but figure failed: {exc}")
            self._fill_table(r)
            self._status.setText(
                f"Done ({r.mode}): {r.n_matched}/{r.n_theoretical} matched, "
                f"rms={r.rms_px:.3f} px. Compare or Send."
            )

        self._host._run_async(work, label=f"Orient. peaks ({sc.name})", on_done=on_done)

    def _compare(self) -> None:
        if self._sc is None or self._host._busy:
            return
        if self._result is None:
            self._status.setText("Run Orient. peaks first.")
            return
        sc = self._sc

        def work():
            if sc.indexing_result is None:
                E.index_bvm(sc, log=self._host._console.log, make_figure=False)
            import orientation_peaks as op
            info = op.compare_to_indexing_result(self._result, sc.indexing_result)
            fig = op.make_orientation_peaks_figure(
                self._result,
                title=f"{sc.name} — Orient. vs Index BVM",
                indexing_result=sc.indexing_result,
            )
            return info, fig

        def on_done(r):
            if isinstance(r, Exception):
                self._status.setText(f"Compare error: {r}")
                return
            info, fig = r
            self._compare_info = info
            self._show_fig(fig, owned=True)
            E.register_figure(sc, "orientation_peaks", fig, force=True)
            self._preview_owned = False
            if not info.get("indexing_available"):
                self._lbl_compare.setText("Index BVM result unavailable.")
                return
            self._lbl_compare.setText(
                f"Compare: origin {'=' if info['same_origin'] else '≠'}  "
                f"g1 {'=' if info['same_g1'] else '≠'}  "
                f"g2 {'=' if info['same_g2'] else '≠'}  |  "
                f"hkl agree {info['hkl_agree']}/{info['hkl_common_peaks']} common  |  "
                f"Index inliers={info['index_n_inliers']}  Orient matched={info['orient_n_matched']}"
            )
            self._status.setText("Compare ready — params unchanged until Send.")

        self._host._run_async(work, label=f"Compare Orient/Index ({sc.name})", on_done=on_done)

    def _send(self) -> None:
        if self._sc is None:
            return
        if self._result is None:
            self._status.setText("Run Orient. peaks first.")
            return
        self._collect_params()
        E.apply_orientation_peaks_to_basis_params(
            self._sc, self._result, log=self._host._console.log
        )
        fig = getattr(self, "_preview_fig", None)
        if fig is not None:
            E.register_figure(self._sc, "orientation_peaks", fig, force=True)
            self._preview_owned = False
        self._host._params.reload()
        self._host._update_active_views()
        self._status.setText(
            f"Sent to Fast4D: index_origin={self._sc.params.index_origin} "
            f"g1={self._sc.params.index_g1} g2={self._sc.params.index_g2} "
            f"QR={self._sc.params.qr_rotation:.2f}° "
            f"coord_rot={self._sc.params.coordinate_rotation:.2f}° "
            f"(manual_enabled=True). Open Basis / Strain to verify."
        )

    def _export(self) -> None:
        if self._result is None or self._sc is None:
            self._status.setText("Run Orient. peaks first.")
            return
        sc = self._sc
        if sc.results_dir:
            out = Path(sc.results_dir) / "orientation_peaks"
        else:
            out = Path(sc.braggpeaks_path).resolve().parent / f"{sc.name}_orientation_peaks" \
                if sc.braggpeaks_path else Path.cwd() / "orientation_peaks_export"
        out.mkdir(parents=True, exist_ok=True)
        import orientation_peaks as op
        csv_path = op.export_matches_csv(self._result, out / "matches.csv")
        fig = getattr(self, "_preview_fig", None)
        if fig is not None:
            fig.savefig(out / "overlay.png", dpi=150, bbox_inches="tight")
        self._status.setText(f"Exported → {csv_path.parent}")
