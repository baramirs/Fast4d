"""fast4d.qt_quickstart — Quick Start guide dialog.

A 5-step wizard that appears on first launch (or from Help → Quick Start Guide)
to orient new users without hiding any functionality from experienced ones.

The flag ``fast4d/quickstart_shown`` is stored in QSettings so the dialog only
auto-pops once per installation.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


_PAGES = [
    # (title, body_html)
    (
        "Welcome to Fast4D",
        """
        <p><b>Fast4D</b> is a desktop application for measuring atomic-scale
        <b>strain and stress</b> in crystalline materials using 4D-STEM electron
        microscopy data.</p>

        <p>It takes your microscope data and turns it into color maps that show
        exactly where and how much a crystal is being stretched or compressed
        at the atomic level — useful for semiconductor research, materials
        characterization, and device analysis.</p>

        <p style="color:#1565C0; font-weight:bold;">
        This guide walks you through the 5 key concepts in ~2 minutes.
        You can reopen it any time from <i>Help &rarr; Quick Start Guide</i>.
        </p>
        """,
    ),
    (
        "Step 1 — What kind of data do you have?",
        """
        <p>Fast4D has <b>two starting points</b>:</p>

        <table cellspacing="8" width="100%">
        <tr>
          <td width="48%" style="background:#E3F2FD; padding:10px; border-radius:6px; vertical-align:top;">
            <b>Path A &mdash; I have a braggpeaks.h5 file</b><br><br>
            The raw 4D dataset was already processed through Bragg peak detection:
            diffraction spot positions were extracted and saved to <code>braggpeaks.h5</code>.
            You work with this file directly &mdash; <b>the raw .mib is not needed</b>.<br><br>
            <span style="color:#1565C0;">&check; Fastest path to strain maps<br>
            &check; No GPU needed<br>
            &check; Most common case in the lab</span>
          </td>
          <td width="4%"></td>
          <td width="48%" style="background:#FFF3E0; padding:10px; border-radius:6px; vertical-align:top;">
            <b>Path B &mdash; I have raw .mib data only</b><br><br>
            No braggpeaks file exists yet. You must first tune the detection
            parameters and run the full Bragg peak detection pass
            (Probe &rarr; 6 Points &rarr; Detection) to produce one.<br><br>
            <span style="color:#E65100;">&oplus; Requires a vacuum scan for the probe<br>
            &oplus; GPU (CUDA) strongly recommended</span>
          </td>
        </tr>
        </table>

        <p style="margin-top:12px;">
        Click <b>Load&hellip;</b> in the Files panel to load your data.
        Fast4D auto-detects which path applies.
        </p>
        """,
    ),
    (
        "Step 2 — The calibration steps",
        """
        <p>After loading, the <b>toolbar icons</b> guide you left &rarr; right.
        Steps 1&ndash;3 are Path B only; steps 4&ndash;11 apply to all paths.</p>

        <table cellspacing="0" cellpadding="5" width="100%"
               style="border-collapse:collapse; font-size:12px;">
          <tr style="background:#BBDEFB;">
            <th align="left" style="padding:5px 8px; width:90px;">Step</th>
            <th align="left" style="padding:5px 8px; width:80px;">Who</th>
            <th align="left" style="padding:5px 8px;">What it does</th>
          </tr>
          <tr style="background:#F7FBFF;">
            <td style="padding:4px 8px;"><b>Probe</b></td>
            <td style="padding:4px 8px; color:#888;">Path B only</td>
            <td style="padding:4px 8px;">Define the electron beam template from a vacuum scan</td>
          </tr>
          <tr style="background:#EEF7FF;">
            <td style="padding:4px 8px;"><b>6 Points</b></td>
            <td style="padding:4px 8px; color:#888;">Path B only</td>
            <td style="padding:4px 8px;">Pick 6 ADF reference points to preview detection quality</td>
          </tr>
          <tr style="background:#F7FBFF;">
            <td style="padding:4px 8px;"><b>Detection</b></td>
            <td style="padding:4px 8px; color:#888;">Path B only</td>
            <td style="padding:4px 8px;">Tune parameters and run Bragg peak detection &rarr; produces braggpeaks.h5</td>
          </tr>
          <tr style="background:#EEF7FF;">
            <td style="padding:4px 8px;"><b>ROI</b></td>
            <td style="padding:4px 8px; color:#1565C0;"><b>All paths</b></td>
            <td style="padding:4px 8px;">Select the calibration region on the ADF image</td>
          </tr>
          <tr style="background:#F7FBFF;">
            <td style="padding:4px 8px;"><b>Origin</b></td>
            <td style="padding:4px 8px; color:#1565C0;"><b>All paths</b></td>
            <td style="padding:4px 8px;">Set the center of the diffraction pattern (zero-beam)</td>
          </tr>
          <tr style="background:#EEF7FF;">
            <td style="padding:4px 8px;"><b>Ellipse</b></td>
            <td style="padding:4px 8px; color:#1565C0;"><b>All paths</b></td>
            <td style="padding:4px 8px;">Correct detector distortion (optional but recommended)</td>
          </tr>
          <tr style="background:#F7FBFF;">
            <td style="padding:4px 8px;"><b>Q Pixel</b></td>
            <td style="padding:4px 8px; color:#1565C0;"><b>All paths</b></td>
            <td style="padding:4px 8px;">Set the reciprocal-space pixel size (&Aring;&minus;&sup1;/px)</td>
          </tr>
          <tr style="background:#EEF7FF;">
            <td style="padding:4px 8px;"><b>Basis</b></td>
            <td style="padding:4px 8px; color:#1565C0;"><b>All paths</b></td>
            <td style="padding:4px 8px;">Define the crystal lattice vectors
            (optional: <b>Index BVM&hellip;</b> proposes g1/g2 from the BVM —
            see <i>Help &rarr; Index BVM guide</i>)</td>
          </tr>
          <tr style="background:#F7FBFF;">
            <td style="padding:4px 8px;"><b>Strain</b></td>
            <td style="padding:4px 8px; color:#1565C0;"><b>All paths</b></td>
            <td style="padding:4px 8px;">Compute strain maps: &epsilon;_xx, &epsilon;_yy, &epsilon;_xy, &theta;</td>
          </tr>
          <tr style="background:#EEF7FF;">
            <td style="padding:4px 8px;"><b>Analysis</b></td>
            <td style="padding:4px 8px; color:#1565C0;"><b>All paths</b></td>
            <td style="padding:4px 8px;">Stress + line/ROI tools via menu
            <b>Tools</b> (Live Line, Live ROI, Set up, Analyse)</td>
          </tr>
        </table>
        """,
    ),
    (
        "Step 3 — The fastest way to get strain maps",
        """
        <p>For most Path A users (braggpeaks.h5 already exists), you need
        <b>3 actions</b>:</p>

        <ol style="line-height:2.0;">
          <li>
            <b>Load&hellip;</b> &mdash; select your <code>.h5</code> or
            <code>braggpeaks.h5</code> file from the Files panel
          </li>
          <li>
            <b>Fill the Parameter Table</b> &mdash; center guess, Q px, basis vectors
            (copy from a previous session with <i>Load calib from h5&hellip;</i>
            under &ldquo;&ctdot; More&rdquo;)
          </li>
          <li>
            <b>&#9658; Compute</b> &mdash; runs the full calibration chain
            and produces strain maps
          </li>
        </ol>

        <p style="background:#E8F5E9; padding:10px; border-radius:6px; color:#1B5E20;">
        <b>Tip:</b> Use <i>Calib = apply</i> (not fit) when you already have good
        parameters from a previous session &mdash; it is much faster.
        </p>

        <p style="background:#E3F2FD; padding:10px; border-radius:6px; color:#0D47A1;">
        <b>Tip:</b> Each toolbar step has a <i>&ldquo;Setting [step] calibration&rdquo;</i>
        button that opens an interactive tool to visually verify each parameter
        before computing.
        </p>
        """,
    ),
    (
        "Step 4 — Finding help as you work",
        """
        <p>Fast4D has several layers of built-in help:</p>

        <ul style="line-height:2.0;">
          <li>
            <b>Tooltips</b> &mdash; hover over any button for a detailed explanation.
          </li>
          <li>
            <b>Console dock</b> (bottom) &mdash; shows all log messages, warnings,
            and step-by-step progress during Compute.
          </li>
          <li>
            <b>Calibration Guide</b> (Help menu) &mdash; quick reference for
            the fit vs apply modes, figure modes, and toolbar verbs.
          </li>
          <li>
            <b>Index BVM Guide</b> (Help menu) &mdash; RANSAC + hkl indexing
            to propose Basis vectors before manual setup.
          </li>
          <li>
            <b>Parameter Table</b> &mdash; hover over any row label to see what
            that parameter does in the pipeline.
          </li>
          <li>
            <b>Status strip</b> (left panel) &mdash; shows calibration status per step:<br>
            &nbsp;&nbsp;&nbsp; &cir; pending &nbsp; &compfn; computed &nbsp;
            &check; done &nbsp; &cross; error
          </li>
        </ul>

        <p style="color:#1565C0; font-weight:bold; margin-top:12px;">
        You are ready to start. Close this guide and click
        <b>Load&hellip;</b> in the Files panel.
        </p>
        """,
    ),
]


class QuickStartDialog(QtWidgets.QDialog):
    """5-page Quick Start wizard. Auto-shown on first launch; reopenable from Help menu."""

    _SETTINGS_KEY = "fast4d/quickstart_shown"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quick Start Guide — Fast4D")
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
            | QtCore.Qt.WindowType.WindowMaximizeButtonHint
        )
        self.resize(680, 520)
        self._page = 0
        self._build()
        self._go(0)

    def _build(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 12)
        lay.setSpacing(10)

        # Progress dots
        dots_row = QtWidgets.QHBoxLayout()
        dots_row.addStretch(1)
        self._dots: list[QtWidgets.QLabel] = []
        for _ in range(len(_PAGES)):
            dot = QtWidgets.QLabel("●")
            dot.setStyleSheet("color:#BBDEFB; font-size:14px;")
            dots_row.addWidget(dot)
            self._dots.append(dot)
        dots_row.addStretch(1)
        lay.addLayout(dots_row)

        # Title
        self._title = QtWidgets.QLabel()
        self._title.setStyleSheet(
            "font-size:17px; font-weight:bold; color:#0D47A1; margin-bottom:4px;"
        )
        self._title.setWordWrap(True)
        lay.addWidget(self._title)

        # Body
        self._body = QtWidgets.QTextBrowser()
        self._body.setOpenExternalLinks(False)
        self._body.setStyleSheet(
            "QTextBrowser { background:#F7FBFF; border:1px solid #90CAF9;"
            " border-radius:5px; padding:8px; font-size:12px; }"
        )
        lay.addWidget(self._body, 1)

        # Navigation
        nav = QtWidgets.QHBoxLayout()
        self._btn_back = QtWidgets.QPushButton("← Back")
        self._btn_back.clicked.connect(lambda: self._go(self._page - 1))
        self._btn_next = QtWidgets.QPushButton("Next →")
        self._btn_next.clicked.connect(lambda: self._go(self._page + 1))
        self._btn_close = QtWidgets.QPushButton("Close")
        self._btn_close.clicked.connect(self.accept)
        self._chk_noshow = QtWidgets.QCheckBox("Don't show on startup")
        self._chk_noshow.setChecked(True)
        nav.addWidget(self._chk_noshow)
        nav.addStretch(1)
        nav.addWidget(self._btn_back)
        nav.addWidget(self._btn_next)
        nav.addWidget(self._btn_close)
        lay.addLayout(nav)

    def _go(self, idx: int) -> None:
        idx = max(0, min(idx, len(_PAGES) - 1))
        self._page = idx
        title, body = _PAGES[idx]
        self._title.setText(f"{idx + 1} / {len(_PAGES)}  —  {title}")
        self._body.setHtml(body)
        self._btn_back.setEnabled(idx > 0)
        last = idx == len(_PAGES) - 1
        self._btn_next.setEnabled(not last)
        self._btn_next.setText("Finish" if last else "Next →")
        for i, dot in enumerate(self._dots):
            dot.setStyleSheet(
                "color:#1976D2; font-size:14px;" if i == idx
                else "color:#BBDEFB; font-size:14px;"
            )

    def accept(self) -> None:
        if self._chk_noshow.isChecked():
            QtCore.QSettings("Fast4D", "app").setValue(self._SETTINGS_KEY, True)
        super().accept()

    # ── class-level helpers ───────────────────────────────────────────────────

    @classmethod
    def should_auto_show(cls) -> bool:
        """True when the user has never seen the guide (or cleared QSettings)."""
        shown = QtCore.QSettings("Fast4D", "app").value(cls._SETTINGS_KEY, False)
        return str(shown).lower() not in ("true", "1")

    @classmethod
    def maybe_show(cls, parent=None) -> None:
        """Show the dialog on first launch only; no-op on subsequent launches."""
        if cls.should_auto_show():
            cls(parent).exec()
