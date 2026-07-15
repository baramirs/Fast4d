"""fast4d.qt_report — Report panel: per-scan tree browser + session analysis.

Browse figures with a filtered tree (Calibrations / Maps / Reports). Only the
selected leaf is resolved into a Figure (lazy — no bulk materialization).

Line / ROI overlays are **not** auto-built into the tree; they appear under
``Reports`` after Live Line / Live ROI → Send to Report (``report_*`` keys).

Legacy derived keys (``line_profiles``, ``maps_with_lines``, …) still show if
already present on disk/RAM, under a Legacy branch — they are never regenerated
just by opening Report.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

import engine as E

# ── figure key taxonomy ───────────────────────────────────────────────────────
FIG_LABELS = {
    "probe": "Probe (4-panel)", "select6": "Detection @ 6 points",
    "detection": "Detection", "roi": "ROI",
    "origin": "Origin (+ residuals)", "ellipse": "Ellipse",
    "q_pixel": "Q-pixel calibration (Update / current)", "basis": "Basis vectors",
    "indexing": "BVM indexing",
    "strain_without_roi": "Strain map", "strain_with_roi": "Strain map",
    "stress_without_roi": "Stress map", "stress_with_roi": "Stress map",
    "line_profiles": "Line profiles (legacy)", "maps_with_lines": "Maps with lines (legacy)",
    "roi_profiles": "ROI stats (legacy)", "maps_with_rois": "Maps with ROIs (legacy)",
    "roi_distribution": "ROI distribution (legacy)", "lines": "Lines overlay (legacy)",
}

# Channel leaves under each strain / stress composite (filterable: type "exx")
_STRAIN_CHANNELS = (
    ("exx", "exx"),
    ("eyy", "eyy"),
    ("exy", "exy"),
    ("orientation", "orientation (θ)"),
)
_STRESS_CHANNELS = (
    ("sxx", "sxx"),
    ("syy", "syy"),
    ("sxy", "sxy"),
)

_CALIB_KEYS = (
    "probe", "select6", "detection", "roi", "origin", "ellipse",
    "q_pixel", "basis", "indexing",
)
_MAP_WITHOUT = ("strain_without_roi", "stress_without_roi")
_MAP_WITH = ("strain_with_roi", "stress_with_roi")
_LEGACY_DERIVED = {
    "line_profiles", "maps_with_lines", "roi_profiles", "maps_with_rois",
    "roi_distribution", "lines",
}

# Session analysis actions (not per-scan leaves)
_SESSION_ACTIONS = [
    ("Strain distribution", "dist"),
    ("Box / Violin by channel", "box"),
    ("Channel correlation (active scan)", "corr"),
    ("PCA of scans", "pca"),
    ("Stress summary (table)", "stress"),
    ("Strain stats (table)", "stats"),
    ("Calibration values (plot)", "calib_plot"),
    ("Calibration values (table)", "calib_table"),
    ("Pixel-wise difference", "pixdiff"),
    ("Repeatability metrics (table)", "pixdiff_table"),
]

_ROLE = QtCore.Qt.ItemDataRole.UserRole


def _is_report_key(key: str) -> bool:
    return (
        key.startswith("report_")
        or key.startswith("line_group_")
        or key.startswith("line_map_")
        or key.startswith("roi_group_")
        or key.startswith("roi_map_")
    )


def _label_for_key(key: str) -> str:
    if key in FIG_LABELS:
        return FIG_LABELS[key]
    if key.startswith("report_line_group_"):
        return f"Grouped lines — {key[len('report_line_group_'):]}"
    if key.startswith("report_line_profiles_"):
        return f"Line profiles — {key[len('report_line_profiles_'):]}"
    if key.startswith("report_line_"):
        return f"Line map — {key[len('report_line_'):]}"
    if key.startswith("report_roi_group_"):
        return f"Grouped ROIs — {key[len('report_roi_group_'):]}"
    if key.startswith("report_roi_profiles_"):
        return f"ROI profiles — {key[len('report_roi_profiles_'):]}"
    if key.startswith("report_roi_"):
        return f"ROI map — {key[len('report_roi_'):]}"
    if key.startswith("line_group_"):
        return f"Grouped line {key[11:]} (legacy)"
    if key.startswith("roi_group_"):
        return f"Grouped ROI {key[10:]} (legacy)"
    return key


def classify_figure_key(key: str) -> str:
    """Return branch id: calib | map_without | map_with | reports | legacy | other."""
    if key in _CALIB_KEYS:
        return "calib"
    if key in _MAP_WITHOUT:
        return "map_without"
    if key in _MAP_WITH:
        return "map_with"
    if _is_report_key(key):
        return "reports"
    if key in _LEGACY_DERIVED:
        return "legacy"
    return "other"


class ReportPanel(QtWidgets.QWidget):
    saveRequested = QtCore.Signal(bool)
    exportPptxRequested = QtCore.Signal()

    def __init__(self, get_scans, get_active_scan=None, parent=None) -> None:
        super().__init__(parent)
        self._get_scans = get_scans
        self._get_active_scan = get_active_scan
        self._fig = None
        self._df = None
        self._canvas = None
        self._tb = None
        self._owned_channel_fig = None  # transient channel panel we must close
        self._suppress = True  # until UI widgets exist
        self._pending_select: tuple[int, str] | None = None  # (scan_idx, key)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # ── filters ──────────────────────────────────────────────────────────
        filt = QtWidgets.QHBoxLayout()
        filt.addWidget(QtWidgets.QLabel("Show:"))
        self._chk_calib = QtWidgets.QCheckBox("Calibrations")
        self._chk_maps = QtWidgets.QCheckBox("Maps")
        self._chk_reports = QtWidgets.QCheckBox("Reports")
        self._chk_legacy = QtWidgets.QCheckBox("Legacy")
        self._chk_session = QtWidgets.QCheckBox("Session")
        for c in (self._chk_calib, self._chk_maps, self._chk_reports,
                  self._chk_legacy, self._chk_session):
            c.setChecked(True)
            filt.addWidget(c)
        self._chk_legacy.setChecked(False)  # hide legacy by default
        self._chk_legacy.setToolTip(
            "Show old auto-built line/ROI figures if they already exist "
            "(not regenerated). Prefer Live → Send → Reports.")
        filt.addSpacing(8)
        filt.addWidget(QtWidgets.QLabel("Filter:"))
        self._filter_text = QtWidgets.QLineEdit()
        self._filter_text.setPlaceholderText("search…")
        self._filter_text.setClearButtonEnabled(True)
        filt.addWidget(self._filter_text, 1)
        lay.addLayout(filt)

        # ── splitter: tree | preview ─────────────────────────────────────────
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = QtWidgets.QWidget()
        left_l = QtWidgets.QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setAnimated(True)
        self._tree.itemSelectionChanged.connect(self._on_tree_selection)
        left_l.addWidget(self._tree, 1)

        # session options (channel / map label) — used when a Session leaf is selected
        sess = QtWidgets.QHBoxLayout()
        sess.addWidget(QtWidgets.QLabel("Channel:"))
        self._sess_channel = QtWidgets.QComboBox()
        for lbl, val in (("ε_yy", "eyy"), ("ε_xx", "exx"), ("ε_xy", "exy"),
                         ("σ_xx", "sxx"), ("σ_yy", "syy"), ("σ_xy", "sxy"), ("ADF", "adf")):
            self._sess_channel.addItem(lbl, val)
        sess.addWidget(self._sess_channel)
        sess.addWidget(QtWidgets.QLabel("Map:"))
        self._sess_label = QtWidgets.QComboBox()
        for key, title in E.ROI_REF_LABELS.items():
            self._sess_label.addItem(title, key)
        sess.addWidget(self._sess_label)
        sess.addStretch(1)
        left_l.addLayout(sess)

        split.addWidget(left)

        right = QtWidgets.QWidget()
        right_l = QtWidgets.QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        self._stack = QtWidgets.QStackedWidget()
        figpage = QtWidgets.QWidget()
        self._fig_host = QtWidgets.QVBoxLayout(figpage)
        self._fig_host.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QtWidgets.QLabel(
            "Select a leaf in the tree — it shows automatically.\n"
            "Calibrations & Maps come from Compute.\n"
            "Reports come from Live Line / Live ROI → Send to Report.")
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color:#888;")
        self._fig_host.addWidget(self._placeholder, 1)
        self._table = QtWidgets.QTableWidget()
        self._stack.addWidget(figpage)
        self._stack.addWidget(self._table)
        right_l.addWidget(self._stack, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 700])
        lay.addWidget(split, 1)

        # ── actions ──────────────────────────────────────────────────────────
        row = QtWidgets.QHBoxLayout()
        for txt, fn, tip in (
            ("Show", self._show_current,
             "Re-render the current selection (usually automatic)."),
            ("Maximize", self._maximize, None),
            ("Export table…", self._export, None),
            ("Export…", lambda: self.exportPptxRequested.emit(),
             "Export a PDF / DOCX / PPTX report from per-channel maps "
             "(no collage cropping)."),
            ("Save", lambda: self.saveRequested.emit(False), None),
            ("Save As…", lambda: self.saveRequested.emit(True), None),
        ):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            if tip:
                b.setToolTip(tip)
            row.addWidget(b)
        row.addStretch(1)
        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet("color:#1565C0; font-size:10px;")
        row.addWidget(self._status)
        lay.addLayout(row)

        # Connect filters only after _tree exists (setChecked would fire early otherwise).
        for c in (self._chk_calib, self._chk_maps, self._chk_reports,
                  self._chk_legacy, self._chk_session):
            c.toggled.connect(self._on_filter_changed)
        self._filter_text.textChanged.connect(self._on_filter_changed)
        self._sess_channel.currentIndexChanged.connect(self._on_session_opts)
        self._sess_label.currentIndexChanged.connect(self._on_session_opts)

        self._suppress = False
        self.refresh()

    # ── public API ────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Rebuild the tree from available figure keys (no bulk Figure load)."""
        self._suppress = True
        try:
            self._rebuild_tree()
        finally:
            self._suppress = False
        if self._pending_select is not None:
            si, key = self._pending_select
            self._pending_select = None
            self.select_figure(si, key)
        else:
            self._auto_select_first_leaf()

    def select_figure(self, scan_idx: int, key: str) -> None:
        """Select a figure leaf after Send to Report (or external jump)."""
        root = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            scan_item = root.child(i)
            payload = scan_item.data(0, _ROLE) or {}
            if payload.get("kind") != "scan" or payload.get("scan_idx") != scan_idx:
                continue
            leaf = self._find_leaf(scan_item, key)
            if leaf is not None:
                self._tree.setCurrentItem(leaf)
                self._tree.scrollToItem(leaf)
                return
        # Not in tree yet (filters?) — retry on next refresh
        self._pending_select = (scan_idx, key)

    # ── tree build ────────────────────────────────────────────────────────────
    def _on_filter_changed(self, *_args) -> None:
        if self._suppress:
            return
        self._suppress = True
        try:
            self._rebuild_tree()
        finally:
            self._suppress = False

    def _rebuild_tree(self) -> None:
        prev = self._current_payload()
        self._tree.clear()
        scans = self._get_scans() or []
        needle = (self._filter_text.text() or "").strip().lower()
        show_calib = self._chk_calib.isChecked()
        show_maps = self._chk_maps.isChecked()
        show_reports = self._chk_reports.isChecked()
        show_legacy = self._chk_legacy.isChecked()
        show_session = self._chk_session.isChecked()

        active = self._get_active_scan() if self._get_active_scan else None

        for si, sc in enumerate(scans):
            keys = list(E.list_figure_keys(sc))
            # Index BVM: show under Calibrations once indexing_result exists
            if (getattr(sc, "indexing_result", None) is not None
                    and "indexing" not in keys):
                keys.append("indexing")
            by = {
                "calib": [], "map_without": [], "map_with": [],
                "reports": [], "legacy": [], "other": [],
            }
            for k in keys:
                by[classify_figure_key(k)].append(k)

            scan_item = QtWidgets.QTreeWidgetItem([sc.name])
            scan_item.setData(0, _ROLE, {"kind": "scan", "scan_idx": si})
            font = scan_item.font(0)
            font.setBold(True)
            scan_item.setFont(0, font)
            self._tree.addTopLevelItem(scan_item)

            def add_branch(parent, title: str, key_list: list[str], *, force=False):
                if not key_list and not force:
                    return None
                br = QtWidgets.QTreeWidgetItem([title])
                br.setData(0, _ROLE, {"kind": "branch"})
                parent.addChild(br)
                for k in key_list:
                    lbl = _label_for_key(k)
                    if needle and needle not in lbl.lower() and needle not in k.lower():
                        continue
                    leaf = QtWidgets.QTreeWidgetItem([lbl])
                    leaf.setData(0, _ROLE, {
                        "kind": "figure", "scan_idx": si, "key": k,
                    })
                    br.addChild(leaf)
                if br.childCount() == 0 and not force:
                    parent.removeChild(br)
                    return None
                return br

            def add_map_branch(parent, title: str, key_list: list[str]):
                if not key_list:
                    return None
                br = QtWidgets.QTreeWidgetItem([title])
                br.setData(0, _ROLE, {"kind": "branch"})
                parent.addChild(br)
                for k in key_list:
                    lbl = _label_for_key(k)
                    if k.startswith("strain"):
                        channels = _STRAIN_CHANNELS
                    elif k.startswith("stress"):
                        channels = _STRESS_CHANNELS
                    else:
                        channels = ()
                    map_hit = (not needle
                               or needle in lbl.lower()
                               or needle in k.lower())
                    ch_hits = [
                        (ch, ch_lbl) for ch, ch_lbl in channels
                        if (not needle
                            or needle in ch.lower()
                            or needle in ch_lbl.lower()
                            or map_hit)
                    ]
                    # Narrow to exact channel hits when filter is a channel name
                    if needle and not map_hit:
                        ch_hits = [
                            (ch, ch_lbl) for ch, ch_lbl in channels
                            if needle in ch.lower() or needle in ch_lbl.lower()
                        ]
                    if needle and not map_hit and not ch_hits:
                        continue
                    if not needle:
                        ch_hits = list(channels)

                    map_item = QtWidgets.QTreeWidgetItem([lbl])
                    map_item.setData(0, _ROLE, {
                        "kind": "figure", "scan_idx": si, "key": k,
                    })
                    br.addChild(map_item)
                    roi_label = "with_roi" if "with_roi" in k else "without_roi"
                    for ch, ch_lbl in ch_hits:
                        leaf = QtWidgets.QTreeWidgetItem([ch_lbl])
                        leaf.setData(0, _ROLE, {
                            "kind": "channel",
                            "scan_idx": si,
                            "map_key": k,
                            "channel": ch,
                            "label": roi_label,
                        })
                        map_item.addChild(leaf)
                    if needle and ch_hits and not map_hit:
                        map_item.setExpanded(True)
                if br.childCount() == 0:
                    parent.removeChild(br)
                    return None
                return br

            if show_calib:
                add_branch(scan_item, "Calibrations", by["calib"])
            if show_maps:
                maps = QtWidgets.QTreeWidgetItem(["Maps"])
                maps.setData(0, _ROLE, {"kind": "branch"})
                scan_item.addChild(maps)
                add_map_branch(maps, "Theoretical reference (without ROI)",
                               by["map_without"])
                add_map_branch(maps, "Experimental reference (with ROI)",
                               by["map_with"])
                if maps.childCount() == 0:
                    scan_item.removeChild(maps)
            if show_reports:
                add_branch(scan_item, "Reports", by["reports"] + by["other"],
                           force=False)
            if show_legacy:
                add_branch(scan_item, "Legacy derived", by["legacy"])

            # Expand active scan (or first) so the tree is usable immediately
            if (active is not None and sc is active) or (active is None and si == 0):
                scan_item.setExpanded(True)
                for j in range(scan_item.childCount()):
                    scan_item.child(j).setExpanded(True)

        if show_session:
            sess = QtWidgets.QTreeWidgetItem(["Session analysis"])
            sess.setData(0, _ROLE, {"kind": "session_root"})
            font = sess.font(0)
            font.setBold(True)
            sess.setFont(0, font)
            self._tree.addTopLevelItem(sess)
            for title, action in _SESSION_ACTIONS:
                if needle and needle not in title.lower():
                    continue
                leaf = QtWidgets.QTreeWidgetItem([title])
                leaf.setData(0, _ROLE, {"kind": "session", "action": action})
                sess.addChild(leaf)
            sess.setExpanded(False)

        if prev:
            self._restore_payload(prev)

    def _find_leaf(self, parent: QtWidgets.QTreeWidgetItem, key: str):
        for i in range(parent.childCount()):
            ch = parent.child(i)
            payload = ch.data(0, _ROLE) or {}
            if payload.get("kind") == "figure" and payload.get("key") == key:
                return ch
            found = self._find_leaf(ch, key)
            if found is not None:
                return found
        return None

    def _current_payload(self) -> dict | None:
        it = self._tree.currentItem()
        if it is None:
            return None
        return it.data(0, _ROLE)

    def _restore_payload(self, payload: dict) -> None:
        kind = payload.get("kind")
        root = self._tree.invisibleRootItem()

        def walk(item):
            for i in range(item.childCount()):
                ch = item.child(i)
                p = ch.data(0, _ROLE) or {}
                if kind == "figure" and p.get("kind") == "figure":
                    if (p.get("scan_idx") == payload.get("scan_idx")
                            and p.get("key") == payload.get("key")):
                        self._tree.setCurrentItem(ch)
                        return True
                if kind == "channel" and p.get("kind") == "channel":
                    if (p.get("scan_idx") == payload.get("scan_idx")
                            and p.get("map_key") == payload.get("map_key")
                            and p.get("channel") == payload.get("channel")):
                        self._tree.setCurrentItem(ch)
                        return True
                if kind == "session" and p.get("kind") == "session":
                    if p.get("action") == payload.get("action"):
                        self._tree.setCurrentItem(ch)
                        return True
                if walk(ch):
                    return True
            return False

        walk(root)

    def _auto_select_first_leaf(self) -> None:
        root = self._tree.invisibleRootItem()

        def first_figure(item):
            for i in range(item.childCount()):
                ch = item.child(i)
                p = ch.data(0, _ROLE) or {}
                if p.get("kind") in ("figure", "channel"):
                    return ch
                found = first_figure(ch)
                if found is not None:
                    return found
            return None

        leaf = first_figure(root)
        if leaf is not None:
            self._tree.setCurrentItem(leaf)
        else:
            self._set_figure(None)
            self._status.setText("No figures yet — run Compute or Send from Live tools.")

    # ── selection → show ──────────────────────────────────────────────────────
    def _on_tree_selection(self) -> None:
        if self._suppress:
            return
        self._show_current()

    def _on_session_opts(self, *_args) -> None:
        if self._suppress:
            return
        p = self._current_payload() or {}
        if p.get("kind") == "session":
            self._show_current()

    def _show_current(self) -> None:
        scans = self._get_scans() or []
        payload = self._current_payload()
        if not payload or payload.get("kind") in ("scan", "branch", "session_root"):
            return
        if payload.get("kind") == "figure":
            self._show_figure(scans, payload.get("scan_idx", -1), payload.get("key"))
        elif payload.get("kind") == "channel":
            self._show_channel(scans, payload)
        elif payload.get("kind") == "session":
            self._show_session(scans, payload.get("action"))

    def _resolve_indexing_figure(self, scan):
        fig = E.resolve_figure(scan, "indexing")
        if fig is not None:
            return fig
        result = getattr(scan, "indexing_result", None)
        if result is None:
            return None
        try:
            import bvm_indexing as bix
            fig = bix.make_indexing_figure(
                result, title=f"{scan.name} — BVM indexing")
            E.register_figure(scan, "indexing", fig, force=True)
            return fig
        except Exception:
            return None

    def _show_figure(self, scans: list, scan_idx: int, key: str | None) -> None:
        if key is None or not (0 <= scan_idx < len(scans)):
            self._status.setText("Invalid selection.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            sc = scans[scan_idx]
            if key == "indexing":
                fig = self._resolve_indexing_figure(sc)
            else:
                fig = E.resolve_figure(sc, key)
            if fig is None:
                self._status.setText(f"No figure for «{key}».")
                self._set_figure(None)
                return
            self._df = None
            self._set_figure(fig, owned=False)
            self._status.setText(f"{sc.name} · {_label_for_key(key)}")
        except Exception as exc:
            self._status.setText(f"Error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _show_channel(self, scans: list, payload: dict) -> None:
        si = int(payload.get("scan_idx", -1))
        ch = payload.get("channel")
        label = payload.get("label") or "without_roi"
        map_key = payload.get("map_key") or ""
        if not ch or not (0 <= si < len(scans)):
            self._status.setText("Invalid channel selection.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            sc = scans[si]
            title = (f"{sc.name} — {ch} — "
                     f"{E.roi_ref_label(label)}")
            fig = E.build_channel_panel_figure(sc, ch, label, title=title)
            if fig is None:
                self._status.setText(f"No {ch} map for «{map_key}».")
                self._set_figure(None)
                return
            self._df = None
            self._set_figure(fig, owned=True)
            self._status.setText(f"{sc.name} · {map_key} · {ch}")
        except Exception as exc:
            self._status.setText(f"Error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _show_session(self, scans: list, action: str | None) -> None:
        if not action:
            return
        if not scans:
            self._status.setText("No scans loaded.")
            return
        label = self._sess_label.currentData() or "without_roi"
        channel = self._sess_channel.currentData()
        restricted = False
        grouped = action in ("dist", "box", "pca", "stress", "stats", "calib_plot", "calib_table")
        if grouped and not E.get_analysis_scope().shared_stats:
            active = self._get_active_scan() if self._get_active_scan else None
            if active is not None:
                scans = [active]
                restricted = True
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            import analysis as A
            self._df = None
            if action == "dist":
                ch = channel if channel in ("eyy", "exx", "exy") else "eyy"
                self._set_figure(A.distribution_figure(scans, channel=ch, label=label))
            elif action == "box":
                self._set_figure(A.boxplot_figure(scans, label=label, kind="violin"))
            elif action == "corr":
                active = self._get_active_scan() if self._get_active_scan else scans[0]
                self._set_figure(A.correlation_figure(active, label))
            elif action == "pca":
                self._set_figure(A.pca_figure(scans, label))
            elif action == "stress":
                c11, c12, c44 = scans[0].params.stress_constants_gpa()
                self._set_table(A.stress_summary_table(
                    scans, c11_gpa=c11, c12_gpa=c12, c44_gpa=c44, label=label))
            elif action == "stats":
                self._set_table(A.cross_scan_stats(scans, label))
            elif action == "calib_plot":
                cols = E.calibration_numeric_columns(scans)
                value = cols[0] if cols else None
                self._set_figure(E.build_calibration_value_figure(scans, value))
            elif action == "calib_table":
                self._set_table(E.calibration_values_table(scans))
            elif action == "pixdiff":
                if len(scans) < 2:
                    self._status.setText("Need ≥2 scans for pixel difference.")
                    return
                self._set_figure(E.build_pixel_difference_figure(
                    scans[0], scans[1], channel, label, drift_correct=False))
            elif action == "pixdiff_table":
                self._set_table(E.pixel_difference_table(
                    scans, channel, label, reference_idx=0, drift_correct=False))
            else:
                self._status.setText(f"Unknown session action: {action}")
                return
            note = " — active file only" if restricted else ""
            self._status.setText(f"Session · {action}{note}")
        except Exception as exc:
            self._status.setText(f"Error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ── display helpers ───────────────────────────────────────────────────────
    def _clear_canvas(self) -> None:
        if self._tb is not None:
            self._fig_host.removeWidget(self._tb)
            self._tb.setParent(None)
            self._tb.deleteLater()
            self._tb = None
        if self._canvas is not None:
            self._fig_host.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None
        if self._owned_channel_fig is not None:
            try:
                E.close_figure(self._owned_channel_fig)
            except Exception:
                pass
            self._owned_channel_fig = None

    def _set_figure(self, fig, *, owned: bool = False) -> None:
        # Live Qt canvas (not savefig thumbnail) — strain/stress composites with
        # edited colorbars often fail bbox_inches='tight' and looked blank before.
        self._clear_canvas()
        self._fig = fig
        self._stack.setCurrentIndex(0)
        if fig is None:
            self._placeholder.setVisible(True)
            return
        if owned:
            self._owned_channel_fig = fig
        self._placeholder.setVisible(False)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from qt_widgets import safe_nav_toolbar
        self._canvas = FigureCanvasQTAgg(fig)
        self._tb = safe_nav_toolbar(self._canvas, self)
        self._fig_host.addWidget(self._tb)
        self._fig_host.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def _set_table(self, df) -> None:
        self._df = df
        self._clear_canvas()
        self._fig = None
        t = self._table
        t.clear()
        cols = list(df.columns) if df is not None else []
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels([str(c) for c in cols])
        t.setRowCount(0 if df is None else len(df))
        if df is not None:
            for r in range(len(df)):
                for c, col in enumerate(cols):
                    it = QtWidgets.QTableWidgetItem(str(df.iloc[r, c]))
                    it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                                | QtCore.Qt.ItemFlag.ItemIsSelectable)
                    t.setItem(r, c, it)
        t.resizeColumnsToContents()
        self._stack.setCurrentIndex(1)

    def _maximize(self) -> None:
        if self._fig is None:
            self._status.setText("Maximize works on a figure view.")
            return
        # Preview already owns a FigureCanvasQTAgg on self._fig — a second canvas
        # on the same Figure breaks. Snapshot to PNG for the floating window.
        import tempfile
        from pathlib import Path
        from qt_widgets import FigureDialog, _is_visible_dialog
        path = Path(tempfile.gettempdir()) / "fast4d_report_maximize.png"
        try:
            self._fig.savefig(path, dpi=150)
        except Exception:
            try:
                self._fig.savefig(path, dpi=150, bbox_inches="tight")
            except Exception as exc:
                self._status.setText(f"Maximize failed: {exc}")
                return
        host = self.window()
        if host is not None and not hasattr(host, "_figure_windows"):
            host._figure_windows = []
        if host is not None:
            host._figure_windows = [d for d in host._figure_windows if _is_visible_dialog(d)]
        dlg = FigureDialog.from_png(str(path), host, "Report figure")
        if host is not None:
            host._figure_windows.append(dlg)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _export(self) -> None:
        if self._df is None:
            self._status.setText("Export works on a table view.")
            return
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
