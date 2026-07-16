# Fast4D — Release notes draft (next version)

**Audience:** GitHub PR / Release / CHANGELOG  
**Language:** English  
**Scope:** Workflow, memory, and Report improvements merged into `main` (2026-07-13 … 2026-07-15)  
**Status:** Local merge on `main`; ahead of `origin/main` (not pushed yet)

> **Public build note:** Experimental **BVM indexing** UIs (Index BVM, Orient. peaks, Plugin menu) are **not exposed** in this release. Installers from GitHub get the features below only. Indexing code may exist in the tree for development; enable only with `FAST4D_ENABLE_INDEXING_UI=1` (not documented for end users).

---

## Highlights

| Area | One-liner |
|------|-----------|
| **RAM lifecycle** | Release datacube after Bragg save; optional CPU streaming; Free RAM can drop reloadable peaks |
| **Report** | Auto-show, tree browser, PDF/DOCX/PPTX export without collage cropping |
| **Tools menu** | Live Line / Live ROI / Set up Lines & ROI / Analyse moved under **Tools** |
| **UX** | Modeless figures; View → Tools → Settings → Help; strain theta clim respects user range |
| **EMD / virtual images** | More robust tutorial EMD load; save ADF/BF into the same `.h5` |
| **Crystal (Q-pixel)** | Optional CIF load for Q-pixel calibration crystal |

---

## Included commits (onto `origin/main`)

```text
6df0bed  feat: release datacube after Bragg save + stream Path A + reclaim peaks
9a802f8  feat: Index BVM GUI, Report auto-show, and UX polish
3b0bb30  feat: Report tree/export, Tools menu, and Q-pixel origin guard
bff097f  feat: CIF indexing, Orient. peaks, and Plugin menu for BVM indexers
daf52c8  Merge branch 'peak-indexer-notebook' into main
```

*(Commit messages mention indexing work that shipped in the same merge; the **public UI** for indexing is gated off for this release.)*

---

## What's new

### RAM lifecycle (release after save · stream · reclaim)

Large memmapped datacubes no longer need to stay resident after Bragg peaks are saved. Calibration and strain (Path A) run on compact braggpeaks only.

- **Release after save:** After peaks are detected and written to `braggpeaks_path`, the memmap is closed and `state.datacube` is cleared. On save failure the cube is kept for retry.
- **CPU streamed detection:** Optional batch detection via `bragg_stream` so peak RAM never holds the cube and a full `PointListArray` together. GPU paths (`CUDA` / `CUDA_batched`) remain full-scan for throughput. Force streaming with `FAST4D_STREAM_BRAGG=1`.
- **Disk reclaim:** Batch and **Free RAM** can drop in-memory peaks when a path exists; peaks reload on demand via `ensure_braggpeaks_for_calibration`.
- **Science unchanged:** detection thresholds, template, subpixel, and calib/strain math are untouched.

### Report

- **Auto-show:** selecting a view renders it; no Refresh ritual (lazy; does not preload every map into the panel).
- **Tree browser:** per-scan tree with filters (Calibrations / Maps / Reports / Legacy / Session); maps split into Theoretical (no ROI) vs Experimental (with ROI).
- **Export:** Report → Export… → PDF / DOCX / PPTX with full figures (no collage “scissors”); depends on `python-docx`.

### Tools menu and Analysis

- Menu order: **View → Tools → Settings → Help**.
- **Tools** hosts Live Line Profile, Live ROI Profile, Set up Lines & ROI, Analyse (file), Analysis (all).
- These actions were removed from the Analysis step toolbar and the bottom ∑ Analysis button.

### UX polish

- Modeless figure windows (several open at once).
- Strain orientation (theta) colorbar respects the user vrange (no auto-widen).
- Q-pixel origin guard: refuse Q calibration without a valid origin.
- Crystal editor / Q-pixel: optional **Load CIF…** for the calibration crystal (`pymatgen`).

### EMD load and ADF/BF save

- More robust DataCube path discovery for tutorial EMD/HDF5 (`tree=True` then `tree=False` fallback).
- Virtual images can save ADF/BF into the **same** loaded `.h5` (`mode='ao'` + `emdpath`).

---

## Test plan

### Automated (`py4dstem-01419`)

```text
pytest tests/test_report_tree.py tests/test_report_export.py -q
pytest tests/test_bragg_stream_finalize.py tests/test_free_memory.py tests/test_drop_braggpeaks_reload.py -q
pytest tests/test_emd_datapath.py tests/test_vc_save_same_h5.py tests/test_qpixel_origin_guard.py -q
pytest tests/test_crystal_cif.py -q
```

### Manual

1. Launch GUI (`run_gui.bat`) — confirm **no** Plugin menu and **no** Index BVM / Orient. peaks on Basis.
2. Large MIB memmap → Detect → working set drops after Bragg save; Origin→Strain without reloading the cube.
3. Free RAM → optionally drop peaks → next calibration reloads from `.h5`.
4. Report tree + Export PDF/DOCX/PPTX; open several figures modeless.
5. Tools → Live Line / Live ROI; strain theta clim −2°…2°.
6. Optional: tutorial EMD load; ADF/BF save into same `.h5`; Crystal… → Load CIF for Q-pixel.

---

## Key files (public-facing)

**Updated:** `pipeline.py`, `driver.py`, `engine.py`, `bragg_stream.py`, `qt_main.py`, `qt_report.py`, `qt_widgets.py`, `qt_quickstart.py`, `tools/report_export/*`, `requirements.txt`.

**Not exposed in UI this release:** Index BVM / Orient. peaks dialogs and Plugin menu (gated by `FAST4D_ENABLE_INDEXING_UI`).

---

## Notes for maintainers

- Do not advertise indexing features in the public PR/release body until the UI gate is turned on by default.
- Prefer referring to the merged `main` improvements (RAM, Report, Tools, EMD), not private worktree paths.
- Push when ready: `main` is ahead of `origin/main`.
