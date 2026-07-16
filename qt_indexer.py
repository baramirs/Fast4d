"""qt_indexer — IndexerDialog: BVM RANSAC + hkl indexing (Plugin menu).

Modeless dialog opened from **Plugin → Index BVM**. Runs ``engine.index_bvm``,
shows overlay + peak table, and on Send writes ``index_origin/g1/g2`` +
``basis_manual_enabled=True`` into ``scan.params``.
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
    parts = [p for p in (text or "").replace(";", ",").replace("[", "").replace("]", "").split(",")
             if p.strip() != ""]
    if len(parts) != 3:
        return list(default)
    return [int(round(float(p))) for p in parts]


class IndexerDialog(QtWidgets.QDialog):
    """BVM indexing tool: Run → table/overlay → Send to Fast4D / Export."""

    def __init__(self, host, prefer_mode: str | None = None) -> None:
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
        self.setWindowTitle(f"Index BVM — {self._sc.name if self._sc else ''}")
        self.resize(1100, 760)

        lay = QtWidgets.QHBoxLayout(self)
        self._fig_host = QtWidgets.QVBoxLayout()
        fw = QtWidgets.QWidget(); fw.setLayout(self._fig_host)
        lay.addWidget(fw, 1)

        panel = QtWidgets.QWidget(); panel.setMaximumWidth(360)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel(
            "<b>Index BVM</b> — Plugin: RANSAC + hkl (before Basis)"))

        p = self._sc.params if self._sc else E.CalibrationParams()
        self._lbl_crystal = QtWidgets.QLabel("")
        self._lbl_crystal.setWordWrap(True)
        self._lbl_cif_warn = QtWidgets.QLabel("")
        self._lbl_cif_warn.setWordWrap(True)
        self._lbl_cif_warn.setStyleSheet("color:#E65100; font-size:11px;")
        v.addWidget(self._lbl_crystal)
        v.addWidget(self._lbl_cif_warn)
        btn_cif = QtWidgets.QPushButton("Load CIF…")
        btn_cif.setToolTip(
            "Load a crystallographic CIF as the reference crystal "
            "(shared with Q-pixel when cal_crystal=CIF)."
        )
        btn_cif.clicked.connect(self._load_cif)
        v.addWidget(btn_cif)
        self._refresh_crystal_label()

        # Orientation mode: Unknown = lattice + g1/g2 for Basis; Known = absolute hkl
        mode0 = str(prefer_mode or getattr(p, "indexing_orientation_mode", "unknown") or "unknown").lower()
        if mode0 not in ("unknown", "known"):
            mode0 = "unknown"
        row_mode = QtWidgets.QHBoxLayout()
        row_mode.addWidget(QtWidgets.QLabel("Orientation"))
        self._cmb_orient = QtWidgets.QComboBox()
        self._cmb_orient.addItem("Unknown (lattice → Basis)", "unknown")
        self._cmb_orient.addItem("Known (zone + real axes + QR)", "known")
        self._cmb_orient.setCurrentIndex(0 if mode0 == "unknown" else 1)
        self._cmb_orient.setToolTip(
            "Unknown: propose origin/g1/g2 without absolute Miller orientation.\n"
            "Known: require zone axis, real axes H/V, and QR to anchor hkl signs."
        )
        self._cmb_orient.currentIndexChanged.connect(self._on_orient_mode)
        row_mode.addWidget(self._cmb_orient, 1)
        v.addLayout(row_mode)

        self._ed_zone, self._lbl_zone = self._line_row(
            v, "Zone axis [uvw]",
            ",".join(str(int(x)) for x in (p.zone_axis or [1, 1, 0])),
            return_label=True,
        )
        self._ed_zone.setPlaceholderText("optional in Unknown (relative hkl)")
        self._ed_h, self._lbl_h = self._line_row(
            v, "Real axis H (+ry)",
            ",".join(str(int(x)) for x in (p.real_axis_h or [0, 0, -1])),
            return_label=True,
        )
        self._ed_v, self._lbl_v = self._line_row(
            v, "Real axis V (+rx)",
            ",".join(str(int(x)) for x in (p.real_axis_v or [-1, 1, 0])),
            return_label=True,
        )
        self._on_orient_mode()

        tol0 = float(p.indexing_tol_px) if p.indexing_tol_px is not None else float(p.max_peak_spacing)
        self._sp_tol = self._dspin_row(v, "Tolerance (px)", 0.1, 50.0, tol0, 0.1, 2)
        self._sp_seed = self._ispin_row(v, "RANSAC seed", 0, 99999, int(p.indexing_seed))

        row_up = QtWidgets.QHBoxLayout()
        row_up.addWidget(QtWidgets.QLabel("Peak sampling"))
        self._cmb_upsample = QtWidgets.QComboBox()
        for f in (1, 2, 4):
            self._cmb_upsample.addItem(f"{f}×", f)
        self._cmb_upsample.setCurrentIndex(0)
        self._cmb_upsample.setToolTip(
            "Spatial upsampling of the BVM before peak detection (sub-pixel). "
            "1× = default; 2×/4× can refine peak positions."
        )
        row_up.addWidget(self._cmb_upsample, 1)
        v.addLayout(row_up)

        for txt, fn, tip in (
            ("Run indexing", self._run, "Prep upstream calibrations + RANSAC + hkl assign."),
            ("Send to Fast4D", self._send, "Write index_origin/g1/g2 + manual_enabled into params."),
            ("Export CSV/PNG", self._export, "Save indexed table + overlay into the analysis folder."),
        ):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); b.setToolTip(tip)
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

        self._table = QtWidgets.QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["#", "hkl", "d_exp", "d_theo", "Δd%", "I", "res_px", "ok"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMaximumHeight(280)
        self._table.cellClicked.connect(self._on_row_clicked)
        v.addWidget(self._table, 1)

        row_asg = QtWidgets.QHBoxLayout()
        self._btn_g1 = QtWidgets.QPushButton("Set as g1")
        self._btn_g2 = QtWidgets.QPushButton("Set as g2")
        self._btn_g1.clicked.connect(lambda: self._assign_selected("g1"))
        self._btn_g2.clicked.connect(lambda: self._assign_selected("g2"))
        row_asg.addWidget(self._btn_g1); row_asg.addWidget(self._btn_g2)
        v.addLayout(row_asg)

        self._status = QtWidgets.QLabel("Ready — Run indexing to start.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#1565C0; font-size:11px;")
        v.addWidget(self._status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        v.addWidget(bb)
        lay.addWidget(panel)

    # ── widgets ────────────────────────────────────────────────────────────
    def _line_row(self, layout, label, text, *, return_label: bool = False):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(130)
        ed = QtWidgets.QLineEdit(text)
        row.addWidget(lbl); row.addWidget(ed, 1); layout.addLayout(row)
        if return_label:
            return ed, lbl
        return ed

    def _ispin_row(self, layout, label, lo, hi, val):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(130)
        s = QtWidgets.QSpinBox(); s.setRange(int(lo), int(hi)); s.setValue(int(val))
        row.addWidget(lbl); row.addWidget(s, 1); layout.addLayout(row)
        return s

    def _dspin_row(self, layout, label, lo, hi, val, step, dec):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(130)
        s = QtWidgets.QDoubleSpinBox(); s.setDecimals(int(dec))
        s.setRange(float(lo), float(hi)); s.setSingleStep(float(step)); s.setValue(float(val))
        row.addWidget(lbl); row.addWidget(s, 1); layout.addLayout(row)
        return s

    def _orient_mode(self) -> str:
        data = self._cmb_orient.currentData()
        return str(data or "unknown")

    def _on_orient_mode(self, *_args) -> None:
        known = self._orient_mode() == "known"
        for w in (self._ed_h, self._lbl_h, self._ed_v, self._lbl_v):
            w.setEnabled(known)
        self._lbl_zone.setText(
            "Zone axis [uvw]" if known else "Zone axis [uvw] (opt.)"
        )
        tip = (
            "Required for absolute hkl anchoring."
            if known else
            "Optional: enables relative ZOLZ hkl labels (signs not anchored)."
        )
        self._ed_zone.setToolTip(tip)

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
                if p.cal_crystal == "CIF" and not p.cif_path:
                    warn = "cal_crystal=CIF but no cif_path — using default Si until you Load CIF…"
        except Exception as exc:
            crystal = E.CAL_CRYSTALS[E.DEFAULT_CAL_CRYSTAL]
            text = f"Crystal: <b>{crystal.name}</b>  a = {crystal.a_lat:.4f} Å (CIF load failed)"
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
            self._status.setText(f"CIF load failed: {exc}")
            return
        self._sc.params.cal_crystal = "CIF"
        self._sc.params.cif_path = info.path
        try:
            self._host._params.reload()
        except Exception:
            pass
        self._refresh_crystal_label()
        msg = f"Loaded CIF: {Path(info.path).name}  a={info.cal.a_lat:.4f} Å"
        if info.warning:
            msg += " — non-cubic warning (see above)."
        self._status.setText(msg)

    def _collect_params(self) -> None:
        p = self._sc.params
        mode = self._orient_mode()
        p.indexing_orientation_mode = mode
        zone_txt = (self._ed_zone.text() or "").strip()
        if mode == "unknown" and not zone_txt:
            p.zone_axis = []
        else:
            p.zone_axis = _parse_ivec3(zone_txt, [1, 1, 0])
        p.real_axis_h = _parse_ivec3(self._ed_h.text(), [0, 0, -1])
        p.real_axis_v = _parse_ivec3(self._ed_v.text(), [-1, 1, 0])
        p.indexing_tol_px = float(self._sp_tol.value())
        p.indexing_seed = int(self._sp_seed.value())
        if mode == "known" and abs(float(p.qr_rotation)) < 1e-9:
            self._status.setText(
                "Note: QR_rotation≈0° with Known orientation often fails absolute "
                "anchoring — set QR in Basis params, or switch to Unknown."
            )

    # ── figure / table ─────────────────────────────────────────────────────
    def _show_fig(self, fig, *, owned: bool = True):
        if fig is None:
            return
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

    def _fill_table(self, result) -> None:
        peaks = sorted(result.peaks, key=lambda r: r.intensity, reverse=True)
        self._table.setRowCount(len(peaks))
        for r, pk in enumerate(peaks):
            hkl = f"({pk.h} {pk.k} {pk.l})" if pk.ok or (pk.h, pk.k, pk.l) != (0, 0, 0) else "—"
            vals = [
                str(pk.peak_index),
                hkl,
                f"{pk.d_exp:.4f}" if pk.d_exp < 1e6 else "—",
                f"{pk.d_theo:.4f}" if pk.d_theo < 1e6 else "—",
                f"{pk.dd_pct:+.2f}" if pk.dd_pct == pk.dd_pct else "—",
                f"{pk.intensity:.1f}",
                f"{pk.residual_px:.3f}",
                "✓" if pk.ok else "",
            ]
            for c, text in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if c == 0:
                    it.setData(QtCore.Qt.ItemDataRole.UserRole, int(pk.peak_index))
                if pk.peak_index in (result.index_g1, result.index_g2, result.index_origin):
                    it.setBackground(QtGui.QColor("#E3F2FD"))
                self._table.setItem(r, c, it)
        self._table.resizeColumnsToContents()
        mode = result.metrics.get("orientation_mode", "?")
        anchored = result.metrics.get("anchored", False)
        self._lbl_propose.setText(
            f"Proposed: origin={result.index_origin}  g1={result.index_g1} "
            f"{result.metrics.get('g1_hkl_str')}  g2={result.index_g2} "
            f"{result.metrics.get('g2_hkl_str')}  "
            f"inliers={result.n_inliers}/{len(result.peaks)}  "
            f"[{mode}{', anchored' if anchored else ', not anchored'}]"
        )

    def _on_row_clicked(self, row: int, _col: int) -> None:
        it = self._table.item(row, 0)
        if it is None:
            return
        idx = it.data(QtCore.Qt.ItemDataRole.UserRole)
        self._status.setText(f"Selected peak #{idx} — use Set as g1 / g2 to override proposal.")

    def _assign_selected(self, which: str) -> None:
        if self._result is None:
            return
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._status.setText("Select a table row first.")
            return
        it = self._table.item(rows[0].row(), 0)
        idx = int(it.data(QtCore.Qt.ItemDataRole.UserRole))
        pk = next((p for p in self._result.peaks if p.peak_index == idx), None)
        if pk is None:
            return
        if which == "g1":
            self._result.index_g1 = idx
            self._result.g1_px = np_pair(pk.qx, pk.qy)
        else:
            self._result.index_g2 = idx
            self._result.g2_px = np_pair(pk.qx, pk.qy)
        self._fill_table(self._result)
        self._status.setText(f"Assigned peak #{idx} as {which}.")

    # ── actions ────────────────────────────────────────────────────────────
    def _run(self) -> None:
        if self._sc is None or self._host._busy:
            return
        self._collect_params()
        sc = self._sc

        def work():
            up = int(self._cmb_upsample.currentData() or 1)
            return E.index_bvm(
                sc, log=self._host._console.log, make_figure=False, image_upsample=up
            )

        def on_done(r):
            if isinstance(r, Exception):
                self._status.setText(f"Indexing error: {r}")
                return
            self._result = r
            try:
                import bvm_indexing as bix
                fig = bix.make_indexing_figure(r, title=f"{sc.name} — BVM indexing")
                self._show_fig(fig, owned=True)
                # Register immediately so Report → Calibrations shows Index BVM
                E.register_figure(sc, "indexing", fig, force=True)
                self._preview_owned = False
            except Exception as exc:
                self._status.setText(f"Indexed, but figure failed: {exc}")
            self._fill_table(r)
            mode = r.metrics.get("orientation_mode", "?")
            if r.metrics.get("anchored"):
                note = "absolute hkl anchored."
            elif r.metrics.get("relative_hkl"):
                note = "relative hkl only (signs not anchored)."
            else:
                note = "lattice proposal for Basis (no Miller labels)."
            self._status.setText(
                f"Done ({mode}): {r.n_inliers}/{len(r.peaks)} inliers — {note} "
                f"Review table, then Send."
            )

        self._host._run_async(work, label=f"Index BVM ({sc.name})", on_done=on_done)

    def _send(self) -> None:
        if self._sc is None:
            return
        if self._result is None:
            self._status.setText("Run indexing first.")
            return
        self._collect_params()
        E.apply_indexing_to_basis_params(self._sc, self._result, log=self._host._console.log)
        self._sc.indexing_result = self._result
        fig = getattr(self, "_preview_fig", None)
        if fig is not None:
            E.register_figure(self._sc, "indexing", fig, force=True)
            self._preview_owned = False
        self._host._params.reload()
        self._host._update_active_views()
        self._status.setText(
            f"Sent to Fast4D: index_origin={self._sc.params.index_origin} "
            f"g1={self._sc.params.index_g1} g2={self._sc.params.index_g2} "
            f"(manual_enabled=True). Open Basis dialog to verify."
        )

    def _export(self) -> None:
        if self._result is None:
            self._status.setText("Run indexing first.")
            return
        sc = self._sc
        if sc is None:
            return
        if sc.results_dir:
            out = Path(sc.results_dir) / "indexing"
        else:
            out = Path(sc.braggpeaks_path).resolve().parent / f"{sc.name}_indexing" \
                if sc.braggpeaks_path else Path.cwd() / "indexing_export"
        out.mkdir(parents=True, exist_ok=True)
        csv_path = self._result.to_csv(out / "indexed_bvm_peaks_hkl.csv")
        png_path = out / "bvm_indexed_hkl_overlay.png"
        fig = getattr(self, "_preview_fig", None)
        if fig is not None:
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
        self._status.setText(f"Exported:\n{csv_path}\n{png_path if fig else '(no PNG)'}")

    def closeEvent(self, ev) -> None:
        if getattr(self, "_preview_owned", True) and getattr(self, "_preview_fig", None):
            E.close_figure(self._preview_fig)
        self._preview_fig = None
        self._host._maybe_tidy_figures()
        super().closeEvent(ev)


def np_pair(qx: float, qy: float):
    import numpy as np
    return np.array([float(qx), float(qy)], dtype=float)
