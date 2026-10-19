# Fast4D: changes (short summary)

Short list of what was added or fixed, from oldest to newest. The details of each item are in
the commit message (`git log`).

## July 2026: base and memory

- Fast4D GUI and pipeline on py4DSTEM 0.14.19 (pinned in requirements).
- "Create ADF/BF" window with a DP selector and live progress.
- Memory: memmap by default (`FAST4D_FORCE_MEMMAP`), release the heavy data of inactive scans
  and at the end of each file, configurable limit of scans in RAM, batched Bragg detection
  saved to disk (streaming), release the datacube after saving the braggpeaks.
- Interface: reorganized and saved panels, Analysis bar, "Auto-save on Compute".
- Report: figure tree, export to PDF / DOCX / PPTX, Tools menu.
- Q pixel: computes the origin automatically if it is missing before the fit.
- GPLv3 license, SECURITY.md, README.

## September 2026: new tools

- Drift plugin (`drift_estimate.py`): shift between repeated scans (ADF / BF / strain,
  phase correlation), CSV per file; lines and ROIs are placed on the same physical point
  in each file.
- Reproducibility plugin: metrics with and without drift, box / violin plots and PCA.
- Single Line / ROI tool (`qt_measure.py`): lines with width, rotated ROIs, in nm if the scan
  step is known.
- R calibration: scan step (pm/px) read from the Velox EMD (`r_px_pm`); line distances and
  figures in nm.
- File selection for the session analysis (saved in the session JSON) and channel correlation
  over the selected files.
- Vacuum probe cache; mean DP of a ROI.
- Crystal frame of the tensor (`crystal_frame.py`): exx along x_hkl (e.g. [-110]) and eyy along
  y_hkl ([001]); frame card and stress stiffness rotated to those axes.
- External BVM indexing (`bvm_indexing.py`, RANSAC + hkl of the theoretical crystal): orients the
  tensor frame (coordinate_rotation) with all indexed planes; BVM figures with hkl and with
  g1, g2, x, y; warning if g1, g2 are not a primitive pair of the lattice.

## September 2026: strain fixes

- The saved tensor is now the one rotated by coordinate_rotation, with exx / eyy correctly named
  (before, the unrotated one was saved, with exx and eyy swapped).
- Pixels where the fit fails (vacuum) are excluded using the py4DSTEM mask (before, they
  showed up as ~100 %).
- Signs of the g1 / g2 labels resolved with the measured pair (avoids x at 90 deg).
- 'apply' mode: uses the fitted Q pixel, not the seed.
- The GUI no longer freezes between files in a multi-file Compute.

## 28-09-2026: fixes from the engine vs notebook audit

Validated with synthetic data of known truth (Si [110]) and with Scan01. Tests in `tests/`
with a small synthetic generator (`tests/synthetic_si110.py`).

- G1: the "with ROI" reference uses ROI AND fit mask, as py4DSTEM does (before, unfitted pixels
  were included and shifted the whole map by up to -1 % or +100 %).
- G2: removed the ADF "vacuum" mask, which blanked real crystal in fields without vacuum.
- G3: analysis lines and ROIs average only fitted pixels; invalid ones are left empty in the
  CSVs and the valid n is reported (before, they counted as 0 %).
- M1: `ellipse_center` [0, 0] or outside the BVM = use the calibrated origin.
- M2: quality check before applying an ellipse (a/b, center, scale); warnings if the fitted Q
  moves away from the seed or if the ROI is small.
- M3: 'apply' mode applies the fitted ellipse (new output parameter `ellipse_p_fitted`).
- M4: a calibration ROI that cannot be applied raises an error (before, it silently used the
  whole scan).
- M5: strain figures choose the scale per panel (before, they came out blank with uniaxial
  strain).
- D1: warning (WARNING) in detection when minRelativeIntensity > 0.01: with few peaks
  (only 111 / 220) the Q, the BVM indexing and the tensor axis can fail (Scan01 with 0.05:
  Q +57 %, axis rotated 35 deg; with 0.001 correct). Defaults do not change: the user chooses
  them in the detection tab.

## October 2026: robust Q pixel calibration

Validated with a synthetic-truth error benchmark (kept outside the repo) on the GUI's
py4DSTEM 0.14.19, on Windows and in a cloud copy (identical strain to 1.7e-13).

- Global Q pixel search (`q_pixel_global_search`): ring-match grid over +-30 % of the initial
  value, then py4DSTEM local fits; fixes a fit stuck near the seed (unstrained crystal, +4.7 %
  without a warning) and jumps to another ring family (+55 to +92 %, frame -35/-55 deg).
- New parameters `q_search_span` (0.30) and `q_search_tol` (0.01) in the Q-pixel table; the
  calibration stores `q_px_check` (ok / differs / ambiguous).
- If the search reports differs / ambiguous, the GUI asks before computing strain (No stops
  that file); batch runs only log a WARNING. The old +-5 % distance warning is now info.
- Q-pixel dialog "Test" button: "Global search" mode (default) shows the ring-match curve and
  puts the best value in the initial-value box; the previous "Fine sweep 2N+1" remains.
- Strain "with ROI" without the global map computed first no longer raises AttributeError.
- New tests: `test_q_pixel_global_search.py`, `test_qpixel_dialog_global_test.py`,
  `test_q_pixel_search_r2_detected.py` (+ fixture `tests/fixtures/r2/`). Suite: 285 passed,
  2 skipped.
