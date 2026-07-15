"""qt_indexer — IndexerDialog: BVM RANSAC + hkl indexing before Basis setup.

Modeless dialog opened from the Basis toolbar ("Index BVM…"). Runs
``engine.index_bvm``, shows overlay + peak table, and on Send writes
``index_origin/g1/g2`` + ``basis_manual_enabled=True`` into ``scan.params``.
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
        self.setWindowTitle(f"Index BVM — {self._sc.name if self._sc else ''}")
        self.resize(1100, 760)

        lay = QtWidgets.QHBoxLayout(self)
        self._fig_host = QtWidgets.QVBoxLayout()
        fw = QtWidgets.QWidget(); fw.setLayout(self._fig_host)
        lay.addWidget(fw, 1)

        panel = QtWidgets.QWidget(); panel.setMaximumWidth(360)
        v = QtWidgets.QVBoxLayout(panel)
        v.addWidget(QtWidgets.QLabel(
            "<b>Index BVM</b> — RANSAC + hkl (zone axis) before Basis setup"))

        p = self._sc.params if self._sc else E.CalibrationParams()
        crystal = p.cal_crystal_obj()
        v.addWidget(QtWidgets.QLabel(
            f"Crystal: <b>{crystal.name}</b>  a = {crystal.a_lat:.4f} Å"))

        self._ed_zone = self._line_row(v, "Zone axis [uvw]",
                                       ",".join(str(int(x)) for x in (p.zone_axis or [1, 1, 0])))
        self._ed_h = self._line_row(v, "Real axis H (+ry)",
                                    ",".join(str(int(x)) for x in (p.real_axis_h or [0, 0, -1])))
        self._ed_v = self._line_row(v, "Real axis V (+rx)",
                                    ",".join(str(int(x)) for x in (p.real_axis_v or [-1, 1, 0])))

        tol0 = float(p.indexing_tol_px) if p.indexing_tol_px is not None else float(p.max_peak_spacing)
        self._sp_tol = self._dspin_row(v, "Tolerance (px)", 0.1, 50.0, tol0, 0.1, 2)
        self._sp_seed = self._ispin_row(v, "RANSAC seed", 0, 99999, int(p.indexing_seed))

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
    def _line_row(self, layout, label, text):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(130)
        ed = QtWidgets.QLineEdit(text)
        row.addWidget(lbl); row.addWidget(ed, 1); layout.addLayout(row)
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

    def _collect_params(self) -> None:
        p = self._sc.params
        p.zone_axis = _parse_ivec3(self._ed_zone.text(), [1, 1, 0])
        p.real_axis_h = _parse_ivec3(self._ed_h.text(), [0, 0, -1])
        p.real_axis_v = _parse_ivec3(self._ed_v.text(), [-1, 1, 0])
        p.indexing_tol_px = float(self._sp_tol.value())
        p.indexing_seed = int(self._sp_seed.value())

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
        self._lbl_propose.setText(
            f"Proposed: origin={result.index_origin}  g1={result.index_g1} "
            f"{result.metrics.get('g1_hkl_str')}  g2={result.index_g2} "
            f"{result.metrics.get('g2_hkl_str')}  "
            f"inliers={result.n_inliers}/{len(result.peaks)}"
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
            return E.index_bvm(sc, log=self._host._console.log, make_figure=False)

        def on_done(r):
            if isinstance(r, Exception):
                self._status.setText(f"Indexing error: {r}")
                return
            self._result = r
            try:
                import bvm_indexing as bix
                fig = bix.make_indexing_figure(r, title=f"{sc.name} — BVM indexing")
                self._show_fig(fig, owned=True)
            except Exception as exc:
                self._status.setText(f"Indexed, but figure failed: {exc}")
            self._fill_table(r)
            self._status.setText(
                f"Done: {r.n_inliers}/{len(r.peaks)} inliers — review table, then Send."
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
