"""fast4d.qt_loader — the single, smart "Load" dialog.

One dialog that unifies every kind of input the workflow needs:
  • Samples (.mib / .h5)         → braggpeaks auto-detected (``<stem>braggpeaks.h5``)
                                    in the same folder; ADF loaded on demand later.
  • braggpeaks-only files        → Path A scans with no raw.
  • Saved workspace folder(s)    → hydrated (computed) scans for analysis.
  • Shared vacuum scan           → auto-detected (``*vacuum*``); used for the probe.
  • Params / Session JSON        → fills calibration params (template or session).
  • A dedicated PROBE space      → compute the probe from the vacuum file and
                                    preview it (the 4-panel view); applied to all
                                    scans on Load. (ROI-based probe stays in the
                                    Probe step.)

``LoaderDialog.scans`` holds the resulting ``engine.Scan`` list after Load.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

import engine as E
import driver as D
from qt_widgets import ProbeView

# Separate file-dialog filters: raw 4D data vs HDF5/EMD (the user wanted them split).
_RAW_FILTER = "Raw 4D data (*.mib *.dm4 *.dm3 *.raw *.npy *.npz)"
_H5_FILTER = "HDF5 / EMD (*.h5 *.hdf5 *.emd)"
_ALL_FILTER = "All files (*)"
# Valid 4D-STEM data extensions for the vacuum AUTO-search — EXCLUDES .hdr (the Merlin
# header sidecar that sorts before .mib alphabetically and must never be the vacuum).
_DATA_EXTS = {".mib", ".dm4", ".dm3", ".raw", ".h5", ".hdf5", ".emd", ".npy", ".npz"}


def _auto_vacuum(sample_path: str) -> str:
    """First ``*vacuum*`` file next to the sample with a 4D-STEM extension (never a
    .hdr). Prefers .mib/.dm4/raw over .h5; deterministic by name."""
    folder = Path(sample_path).parent
    cands = [c for c in folder.glob("*vacuum*")
             if c.is_file() and c.suffix.lower() in _DATA_EXTS]
    if not cands:
        return ""
    raw = sorted(c for c in cands if c.suffix.lower() in (".mib", ".dm4", ".dm3", ".raw"))
    return str((raw or sorted(cands))[0])


class LoaderDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, existing: list | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags()
                            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
                            | QtCore.Qt.WindowType.WindowMaximizeButtonHint)
        self.setWindowTitle("Load data")
        self.resize(940, 700)
        self.scans: list = []          # result, filled on Load
        self._probe = None             # shared probe (computed from the vacuum)
        self._rows: list[dict] = []    # file-based: {name, raw, bragg}
        self._ws_scans: list = []      # hydrated saved-workspace scans
        self._build()
        if existing:
            for sc in existing:
                self._rows.append({
                    "name": sc.name, "raw": sc.raw_path,
                    "bragg": sc.braggpeaks_path,
                    "h5": getattr(sc, "h5_path", "") or E.find_sidecar_h5(sc.raw_path),
                    "params": getattr(sc, "params_source", ""),
                    "vacuum": getattr(sc, "vacuum_path", "")})
            if existing[0].vacuum_path:
                self._vacuum.setText(existing[0].vacuum_path)
            self._refresh_table()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)

        sg = QtWidgets.QGroupBox(
            "Samples — braggpeaks auto-detected (<stem>braggpeaks.h5), ADF on demand")
        sgl = QtWidgets.QVBoxLayout(sg)
        row = QtWidgets.QHBoxLayout()
        for txt, fn in (("Add samples (.mib/.h5)…", self._add_samples),
                        ("Add braggpeaks-only…", self._add_braggpeaks),
                        ("Add saved workspace…", self._add_workspace),
                        ("Remove selected", self._remove_selected)):
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(1)
        sgl.addLayout(row)
        self._table = QtWidgets.QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Sample (.mib)", "braggpeaks", "Virtual h5 (ADF/BF/DP)",
             "Params (per-file)", "Vacuum (per-file)"])
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        sgl.addWidget(self._table)
        lay.addWidget(sg, 2)

        ig = QtWidgets.QGroupBox("Shared inputs")
        igl = QtWidgets.QGridLayout(ig)
        self._vacuum = QtWidgets.QLineEdit()
        bv = QtWidgets.QPushButton("Browse"); bv.clicked.connect(self._browse_vacuum)
        igl.addWidget(QtWidgets.QLabel("Vacuum (probe):"), 0, 0)
        igl.addWidget(self._vacuum, 0, 1); igl.addWidget(bv, 0, 2)
        self._json = QtWidgets.QLineEdit()
        bj = QtWidgets.QPushButton("Browse"); bj.clicked.connect(self._browse_json)
        igl.addWidget(QtWidgets.QLabel("Params / Session JSON:"), 1, 0)
        igl.addWidget(self._json, 1, 1); igl.addWidget(bj, 1, 2)
        lay.addWidget(ig)

        pg = QtWidgets.QGroupBox("Probe (from the shared vacuum)")
        pgl = QtWidgets.QVBoxLayout(pg)
        prow = QtWidgets.QHBoxLayout()
        b_probe = QtWidgets.QPushButton("Compute probe (vacuum file)")
        b_probe.clicked.connect(self._compute_probe)
        self._probe_on_load = QtWidgets.QCheckBox("Apply probe to all scans on Load")
        self._probe_on_load.setChecked(True)
        prow.addWidget(b_probe); prow.addWidget(self._probe_on_load); prow.addStretch(1)
        pgl.addLayout(prow)
        self._probe_view = ProbeView()
        self._probe_view.setMinimumHeight(180)
        pgl.addWidget(self._probe_view, 1)
        lay.addWidget(pg, 2)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.addButton("Load", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        bb.accepted.connect(self._on_load)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    # ── add / remove ───────────────────────────────────────────────────────────
    def add_paths(self, paths, *, as_braggpeaks: bool = False) -> None:
        """Add file rows (testable without a dialog).

        For a raw sample we auto-pair the light virtual-images sibling ``<stem>.h5``
        (ADF/BF/DP) and the ``<stem>braggpeaks.h5``. The big ``.mib`` is never read for
        preview — only the ``.h5``. A sample missing its ``<stem>.h5`` is flagged (its
        ADF can't be previewed, nor a vacuum region picked, without it)."""
        missing_h5: list[str] = []
        for p in paths:
            p = str(p)
            if as_braggpeaks:
                self._rows.append({"name": Path(p).stem, "raw": "", "bragg": p,
                                   "h5": E.find_sidecar_h5(p), "params": "", "vacuum": ""})
            else:
                bp = E.find_sidecar_braggpeaks(p) or ""
                h5 = E.find_sidecar_h5(p)
                vac = _auto_vacuum(p)               # 4D-STEM ext only (no .hdr)
                self._rows.append({"name": Path(p).stem, "raw": p, "bragg": bp,
                                   "h5": h5, "params": "", "vacuum": vac})
                if not h5:
                    missing_h5.append(Path(p).name)
                if vac and not self._vacuum.text().strip():
                    self._vacuum.setText(vac)       # also seed the shared default
        self._refresh_table()
        if missing_h5:
            QtWidgets.QMessageBox.warning(
                self, "Missing virtual-images .h5",
                "No sibling <stem>.h5 (ADF/BF/DP) was found for:\n  - "
                + "\n  - ".join(missing_h5)
                + "\n\nThe ADF preview comes from that .h5, not the raw .mib, so these "
                  "scans can't show a preview (or pick a vacuum region) until the "
                  "<stem>.h5 exists. Calibration still works, but the raw .mib gets "
                  "loaded (heavy) when first needed.")

    def _add_samples(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add samples", "",
            f"{_RAW_FILTER};;{_H5_FILTER};;{_ALL_FILTER}")
        if paths:
            self.add_paths(paths)

    def _add_braggpeaks(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add braggpeaks", "", "HDF5 (*.h5 *.hdf5);;All (*)")
        if paths:
            self.add_paths(paths, as_braggpeaks=True)

    def _add_workspace(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Add saved workspace(s) — parent or scan folder")
        if not d:
            return
        root = Path(d)
        subs = [str(c) for c in root.iterdir() if c.is_dir() and (c / "data").is_dir()]
        scans = D.hydrate_from_dirs(subs or [str(root)], workspace_root=str(root))
        if scans:
            self._ws_scans.extend(scans)
            self._refresh_table()
        else:
            QtWidgets.QMessageBox.warning(self, "Workspace", "No loadable workspaces here.")

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        n_ws = len(self._ws_scans)
        for r in rows:
            if r < n_ws:
                self._ws_scans.pop(r)
            elif (r - n_ws) < len(self._rows):
                self._rows.pop(r - n_ws)
        self._refresh_table()

    def _refresh_table(self) -> None:
        n_ws = len(self._ws_scans)
        ws = [{"name": s.name, "raw": s.raw_path, "bragg": s.braggpeaks_path,
               "h5": E.scan_h5_path(s), "ws": True} for s in self._ws_scans]
        allrows = ws + [dict(d, ws=False) for d in self._rows]
        self._table.setRowCount(len(allrows))
        for r, d in enumerate(allrows):
            def cell(t, *, ok: bool | None = None):
                it = QtWidgets.QTableWidgetItem(t)
                it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
                if ok is True:
                    it.setForeground(QtGui.QBrush(QtGui.QColor("#2E7D32")))
                elif ok is False:
                    it.setForeground(QtGui.QBrush(QtGui.QColor("#C62828")))
                return it
            self._table.setItem(r, 0, cell(d["name"]))
            self._table.setItem(r, 1, cell(Path(d["raw"]).name if d["raw"] else "—"))
            self._table.setItem(r, 2, cell(("✓ " + Path(d["bragg"]).name) if d["bragg"]
                                           else "✗ — (Path B)", ok=bool(d["bragg"])))
            h5 = d.get("h5", "")
            if d["ws"]:
                self._table.setItem(r, 3, cell("workspace", ok=bool(h5)))
            else:
                fi = r - n_ws
                h5p = self._rows[fi].get("h5", "")
                hbtn = QtWidgets.QPushButton(Path(h5p).name if h5p else "Set h5…")
                hbtn.setToolTip(h5p or "Browse for the virtual-images .h5 (ADF/BF/DP) "
                                "and import calibrations from its metadata if present.")
                hbtn.clicked.connect(lambda _c=False, i=fi: self._pick_row_h5(i))
                self._table.setCellWidget(r, 3, hbtn)
            if d["ws"]:
                self._table.setItem(r, 4, cell("(from workspace)"))
                self._table.setItem(r, 5, cell("(n/a)"))
            else:                                   # per-file params + vacuum buttons
                fi = r - n_ws
                pth = self._rows[fi].get("params", "")
                btn = QtWidgets.QPushButton(Path(pth).name if pth else "Set params…")
                btn.clicked.connect(lambda _c=False, i=fi: self._pick_row_params(i))
                self._table.setCellWidget(r, 4, btn)
                vac = self._rows[fi].get("vacuum", "")
                vbtn = QtWidgets.QPushButton(Path(vac).name if vac else "Set vacuum…")
                vbtn.setToolTip(vac or "Per-file vacuum (overrides the shared default).")
                vbtn.clicked.connect(lambda _c=False, i=fi: self._pick_row_vacuum(i))
                self._table.setCellWidget(r, 5, vbtn)

    def _pick_row_params(self, file_idx: int) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Per-file calibration params (JSON)", "", "JSON (*.json);;All (*)")
        if p and 0 <= file_idx < len(self._rows):
            self._rows[file_idx]["params"] = p
            self._refresh_table()

    def _pick_row_h5(self, file_idx: int) -> None:
        """Browse virtual-images .h5; stash path + optional embedded calibrations."""
        if not (0 <= file_idx < len(self._rows)):
            return
        start = str(Path(self._rows[file_idx]["raw"]).parent) if self._rows[file_idx].get("raw") else ""
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Virtual h5 for '{self._rows[file_idx]['name']}'", start,
            "HDF5 (*.h5 *.hdf5 *.emd);;All (*)")
        if not p:
            return
        self._rows[file_idx]["h5"] = p
        self._refresh_table()

    def _pick_row_vacuum(self, file_idx: int) -> None:
        """Per-file vacuum (so two samples can use two DIFFERENT vacuum files)."""
        if not (0 <= file_idx < len(self._rows)):
            return
        start = str(Path(self._rows[file_idx]["raw"]).parent) if self._rows[file_idx].get("raw") else ""
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Vacuum for '{self._rows[file_idx]['name']}'", start,
            f"{_RAW_FILTER};;{_H5_FILTER};;{_ALL_FILTER}")     # no .hdr
        if p:
            self._rows[file_idx]["vacuum"] = p
            self._refresh_table()

    # ── shared inputs / probe ────────────────────────────────────────────────
    def _browse_vacuum(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Vacuum (shared default)", "",
            f"{_RAW_FILTER};;{_H5_FILTER};;{_ALL_FILTER}")     # no .hdr in the filters
        if p:
            self._vacuum.setText(p)

    def _browse_json(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Params / Session JSON", "", "JSON (*.json);;All (*)")
        if p:
            self._json.setText(p)

    def _compute_probe(self) -> None:
        vac = self._vacuum.text().strip()
        if not vac:
            QtWidgets.QMessageBox.information(
                self, "Probe",
                "Set a vacuum file first.\n(Or pick a vacuum ROI on the sample in "
                "the Probe step after loading.)")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self._probe = E.compute_shared_probe(vac)
            tmp = E.Scan(name="probe")
            tmp.ensure_state()
            tmp.state.probe = self._probe
            self._probe_view.set_figure(E.build_probe_figure(tmp))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Probe", f"Could not compute probe:\n{exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ── build result ──────────────────────────────────────────────────────────
    def build_scans(self) -> list:
        """Assemble the engine.Scan list (testable). Workspace scans first.

        Precedence for calibration params: per-file Params JSON (a row's own) >
        shared Params/Session JSON > defaults. Lets two different samples carry
        individual calibrations in one session.
        """
        scans = list(self._ws_scans)
        has_own = [True] * len(self._ws_scans)      # workspace scans keep their params
        for d in self._rows:
            sc = E.Scan(name=d["name"], raw_path=d["raw"], braggpeaks_path=d["bragg"])
            sc.h5_path = d.get("h5", "")            # light virtual-images sibling (preview)
            sc.vacuum_path = d.get("vacuum", "")    # per-file vacuum (two different vacuums OK)
            try:                                    # self-describing h5 → restore the analysis
                applied = False
                if d.get("h5"):
                    meta = E.read_metadata_h5(d["h5"])
                    if meta:
                        E.apply_metadata_to_scan(sc, meta)
                        applied = True
                if not applied:
                    E.load_metadata_for_scan(sc)
            except Exception:
                pass
            own = False
            if d.get("params"):
                pp = E.params_from_json(d["params"])
                if pp is not None:
                    sc.params = pp
                    sc.params_source = d["params"]
                    own = True
            scans.append(sc)
            has_own.append(own)
        jp = self._json.text().strip()
        if jp:
            try:
                src = E.load_session_json(jp)
            except Exception:
                try:
                    src = E.scans_from_template(jp)
                except Exception:
                    src = []
            if src and not self._rows and not self._ws_scans:
                scans = src                     # JSON defines the whole session
            elif src:
                by = {s.name: s.params for s in src}
                for sc, own in zip(scans, has_own):   # merge by name, per-file wins
                    if not own and sc.name in by:
                        sc.params = by[sc.name]
                        sc.params_source = jp
        vac = self._vacuum.text().strip()
        for sc in scans:
            if vac and not getattr(sc, "vacuum_path", ""):   # per-file vacuum wins; shared fills the rest
                sc.vacuum_path = vac
            if self._probe is not None and self._probe_on_load.isChecked():
                E.apply_probe(sc, self._probe)
        # the shared JSON also carries the line profiles → fix them on the scans here
        # (so the user doesn't have to re-load the same file via the line tool)
        if jp:
            try:
                res = E.load_lines_json(jp, scans)
                if res.get("assigned"):
                    pass  # per-scan lines written onto scans
            except Exception:
                pass
            try:
                E.load_roi_json(jp, scans)   # the same JSON also carries the area ROI
            except Exception:
                pass
        return scans

    def _on_load(self) -> None:
        self.scans = self.build_scans()
        self.accept()
