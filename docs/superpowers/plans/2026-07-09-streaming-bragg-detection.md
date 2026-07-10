# Streaming Bragg-Detection Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in Fast4D-side path that detects Bragg disks scan-position-by-scan-position and writes each position's peaks straight to an HDF5-backed store, instead of relying on py4DSTEM's `PointListArray` — which eagerly allocates a full in-RAM list-of-lists for every scan position *before* any disks are found (`emdfile/pointlistarray.py:89-90`, confirmed in `MEMORY_ARCHITECTURE_REPORT.md` Part IV).

**Architecture:** A new, standalone module `bragg_stream.py` (Fast4D-side, touches neither `py4DSTEM` nor `emdfile` source) built entirely on a public API Fast4D *already calls*: `datacube.find_Bragg_disks(data=(rys, rxs), template=..., **kwargs)` scoped to an arbitrary list of scan positions — the exact same call `detect_selected_bragg_disks_step` (`pipeline.py:1344,1371`) already makes for its 6-point preview, just looped over the whole scan in small batches and written to disk after each batch instead of all at once in RAM. `compute_braggpeaks_step` (`pipeline.py:1380`) is left untouched; this is a new, parallel opt-in path, not a replacement, until it's been validated on real data.

**Tech Stack:** Python 3.10+, `h5py` (already a hard dependency, `requirements.txt`), `numpy`, `py4DSTEM==0.14.19` (public API only).

## Global Constraints

- Do not modify `py4DSTEM`/`emdfile` source — everything lives in new Fast4D-side code, per the report's Part IV risk framing ("upstream change, higher risk... distinct from anything Fast4D directly owns").
- Do not change the numeric result of disk detection — the streaming path must call the identical `find_Bragg_disks` parameters as `compute_braggpeaks_step` uses today, just position-scoped and batched; peak positions/intensities for a given `(ry, rx)` must be bit-identical whether detected via the existing full-scan call or the new streaming call.
- This plan assumes Task 1 of `2026-07-09-quick-wins-memory-release.md` has been merged (adds `tests/conftest.py`, `pytest` in the pinned env). If it hasn't, redo that plan's Task 1 first.
- Tests run inside `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe` (real `py4DSTEM` + `h5py` needed; small synthetic datacubes only — no multi-GB fixtures).

---

### Task 1: `StreamingBraggWriter` — incremental HDF5 peak-list writer

**Files:**
- Create: `bragg_stream.py`
- Test: `tests/test_bragg_stream_writer.py`

**Interfaces:**
- Produces: `StreamingBraggWriter(path: str | Path, r_shape: tuple[int, int])` — context manager; `.write(ry: int, rx: int, qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray) -> None`; on `__exit__`, the file is a valid HDF5 file with one dataset per position under `/peaks/{ry}_{rx}` (each shape `(n_peaks, 3)`, columns `qx, qy, intensity`) plus `/r_shape` = `(R_Ny, R_Nx)`.
- Consumed by: Task 2's `detect_braggpeaks_streaming`.

- [ ] **Step 1: Write the failing test**

`tests/test_bragg_stream_writer.py`:
```python
import numpy as np
import h5py

from bragg_stream import StreamingBraggWriter


def test_writer_creates_one_dataset_per_position(tmp_path):
    path = tmp_path / "stream.h5"
    with StreamingBraggWriter(path, r_shape=(2, 2)) as w:
        w.write(0, 0, qx=np.array([1.0, 2.0]), qy=np.array([1.5, 2.5]), intensity=np.array([10.0, 20.0]))
        w.write(1, 1, qx=np.array([3.0]), qy=np.array([3.5]), intensity=np.array([30.0]))

    with h5py.File(path, "r") as f:
        assert tuple(f["r_shape"][()]) == (2, 2)
        peaks_0_0 = f["peaks"]["0_0"][()]
        assert peaks_0_0.shape == (2, 3)
        np.testing.assert_allclose(peaks_0_0[:, 0], [1.0, 2.0])   # qx
        np.testing.assert_allclose(peaks_0_0[:, 2], [10.0, 20.0])  # intensity
        peaks_1_1 = f["peaks"]["1_1"][()]
        assert peaks_1_1.shape == (1, 3)
        assert "0_1" not in f["peaks"]  # positions never written simply don't exist


def test_writer_handles_zero_peaks_at_a_position(tmp_path):
    path = tmp_path / "stream.h5"
    with StreamingBraggWriter(path, r_shape=(1, 1)) as w:
        w.write(0, 0, qx=np.array([]), qy=np.array([]), intensity=np.array([]))

    with h5py.File(path, "r") as f:
        assert f["peaks"]["0_0"][()].shape == (0, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_bragg_stream_writer.py -v`
Expected: FAIL — `bragg_stream` module does not exist.

- [ ] **Step 3: Implement `StreamingBraggWriter`**

`bragg_stream.py`:
```python
"""
Fast4D-side incremental Bragg-peak writer.

py4DSTEM's own ``PointListArray`` (emdfile/classes/pointlistarray.py:55-90)
eagerly allocates a full list-of-lists covering every scan position before any
disks are found, and has no streaming write path (confirmed in
MEMORY_ARCHITECTURE_REPORT.md Part IV). This module writes one HDF5 dataset
per scan position as detection proceeds, so peak storage never requires the
whole scan's peak list to be resident in RAM at once.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class StreamingBraggWriter:
    """Writes detected peaks to ``path`` one scan position at a time.

    Layout: ``/r_shape`` = (R_Ny, R_Nx); ``/peaks/{ry}_{rx}`` = (n_peaks, 3)
    array of (qx, qy, intensity), one dataset per position that was written.
    """

    def __init__(self, path: str | Path, r_shape: tuple[int, int]) -> None:
        self._path = Path(path)
        self._r_shape = r_shape
        self._file: h5py.File | None = None

    def __enter__(self) -> "StreamingBraggWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "w")
        self._file.create_dataset("r_shape", data=np.array(self._r_shape, dtype=np.int64))
        self._file.create_group("peaks")
        return self

    def write(self, ry: int, rx: int, qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray) -> None:
        if self._file is None:
            raise RuntimeError("StreamingBraggWriter used outside a `with` block.")
        qx = np.asarray(qx, dtype=np.float64)
        qy = np.asarray(qy, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)
        n = qx.shape[0]
        arr = np.empty((n, 3), dtype=np.float64)
        arr[:, 0], arr[:, 1], arr[:, 2] = qx, qy, intensity
        self._file["peaks"].create_dataset(f"{ry}_{rx}", data=arr)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_bragg_stream_writer.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add bragg_stream.py tests/test_bragg_stream_writer.py
git commit -m "feat: add StreamingBraggWriter (incremental HDF5 peak-list storage)"
```

---

### Task 2: `detect_braggpeaks_streaming` — position-batched public-API detection

**Files:**
- Modify: `bragg_stream.py` (append)
- Test: `tests/test_bragg_stream_detect.py`

**Interfaces:**
- Consumes: `StreamingBraggWriter` (Task 1).
- Produces: `detect_braggpeaks_streaming(datacube, template, out_path, *, batch_size=64, log=None, **detect_kwargs) -> Path` — same `**detect_kwargs` accepted by `datacube.find_Bragg_disks` today (`pipeline.py:1416-1418`'s `kwargs`), so it's a drop-in alternative call for the same parameter dict.

- [ ] **Step 1: Write the failing test**

`tests/test_bragg_stream_detect.py`:
```python
import numpy as np
import py4DSTEM
import h5py

from bragg_stream import detect_braggpeaks_streaming


def _tiny_synthetic_datacube():
    # 2x2 scan, 32x32 detector, one bright pixel per pattern so detection has
    # something unambiguous to find without needing a realistic template.
    rng = np.random.default_rng(0)
    data = rng.normal(scale=1.0, size=(2, 2, 32, 32)).astype(np.float32)
    for ry in range(2):
        for rx in range(2):
            data[ry, rx, 16, 16] = 500.0  # single strong disk at center
    return py4DSTEM.DataCube(data)


def test_streaming_matches_full_scan_find_bragg_disks(tmp_path):
    dc = _tiny_synthetic_datacube()
    template = np.zeros((32, 32), dtype=np.float32)
    template[15:18, 15:18] = 1.0  # crude disk-shaped probe kernel

    kwargs = dict(minAbsoluteIntensity=100.0, minPeakSpacing=5, edgeBoundary=2, subpixel="none")

    # Reference: py4DSTEM's own full-scan public API (what compute_braggpeaks_step calls today).
    reference = dc.find_Bragg_disks(template=template, **kwargs)

    out_path = tmp_path / "streamed.h5"
    detect_braggpeaks_streaming(dc, template, out_path, batch_size=1, **kwargs)

    with h5py.File(out_path, "r") as f:
        for ry in range(2):
            for rx in range(2):
                streamed_peaks = f["peaks"][f"{ry}_{rx}"][()]
                ref_pl = reference.get_vectors(ry, rx, name="", subpixel="none") if hasattr(reference, "get_vectors") else reference.cal[ry, rx]
                assert streamed_peaks.shape[0] == len(ref_pl["qx"])
                if streamed_peaks.shape[0]:
                    np.testing.assert_allclose(sorted(streamed_peaks[:, 0]), sorted(ref_pl["qx"]), atol=1e-3)
```

> **Note for the implementer:** py4DSTEM's exact `BraggVectors` accessor API (`get_vectors` vs. `.cal[ry, rx]` vs. `.raw[ry, rx]`) was not pinned down during the investigation behind this plan — confirm the correct accessor against the installed `py4DSTEM==0.14.19` (`braggvectors/braggvectors.py`) before trusting this test's reference-comparison lines, and adjust them to whatever that version's public accessor actually is. This is the one open verification step in this plan; everything else here was grounded against real file:line citations.

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_bragg_stream_detect.py -v`
Expected: FAIL — `detect_braggpeaks_streaming` does not exist yet.

- [ ] **Step 3: Implement `detect_braggpeaks_streaming`**

Append to `bragg_stream.py`:
```python
from typing import Callable

from bragg_stream import StreamingBraggWriter  # (already in this file; shown for clarity)


def detect_braggpeaks_streaming(
    datacube,
    template,
    out_path: str | Path,
    *,
    batch_size: int = 64,
    log: Callable[[str], None] | None = None,
    **detect_kwargs,
) -> Path:
    """Detect Bragg disks in small position-batches and stream results to disk.

    Uses the same public, position-scoped call Fast4D already makes for its
    6-point preview (``pipeline.py:1344,1371``:
    ``datacube.find_Bragg_disks(data=(rys, rxs), template=..., **kwargs)``),
    just looped over the whole scan in ``batch_size``-position chunks instead
    of either 6 positions (preview) or the whole scan at once
    (``compute_braggpeaks_step``, ``pipeline.py:1416``). Only ``batch_size``
    positions' worth of detected peaks are ever held in RAM at a time.
    """
    out_path = Path(out_path)
    r_ny, r_nx = datacube.R_Ny, datacube.R_Nx
    all_positions = [(ry, rx) for ry in range(r_ny) for rx in range(r_nx)]

    with StreamingBraggWriter(out_path, r_shape=(r_ny, r_nx)) as writer:
        for start in range(0, len(all_positions), batch_size):
            batch = all_positions[start:start + batch_size]
            rys = [p[0] for p in batch]
            rxs = [p[1] for p in batch]
            result = datacube.find_Bragg_disks(data=(rys, rxs), template=template, **detect_kwargs)
            for ry, rx in batch:
                pl = result.get_vectors(ry, rx, name="", subpixel=detect_kwargs.get("subpixel", "none")) \
                    if hasattr(result, "get_vectors") else result.cal[ry, rx]
                writer.write(ry, rx, qx=np.asarray(pl["qx"]), qy=np.asarray(pl["qy"]),
                             intensity=np.asarray(pl["intensity"]))
            if log is not None:
                log(f"Streaming Bragg detection: {min(start + batch_size, len(all_positions))}/{len(all_positions)} positions done.")
    return out_path
```

> **Note for the implementer:** the `result.get_vectors(...)`/`result.cal[ry, rx]` accessor line duplicates the same open verification flagged in Step 1 — pin down the correct `BraggVectors` read API for 0.14.19 once, then use it consistently in both this function and its test.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_bragg_stream_detect.py -v`
Expected: 1 passed (after the accessor-API verification note above is resolved).

- [ ] **Step 5: Commit**

```bash
git add bragg_stream.py tests/test_bragg_stream_detect.py
git commit -m "feat: add detect_braggpeaks_streaming (position-batched, disk-backed detection)"
```

---

### Task 3: Reader — reconstruct a `BraggVectors`-compatible view from the streamed file

**Files:**
- Modify: `bragg_stream.py` (append)
- Test: `tests/test_bragg_stream_reader.py`

**Interfaces:**
- Consumes: the HDF5 layout written by Task 1/2 (`/r_shape`, `/peaks/{ry}_{rx}`).
- Produces: `read_streamed_peaks(path: str | Path, ry: int, rx: int) -> dict` returning `{"qx": ndarray, "qy": ndarray, "intensity": ndarray}` for one position, read on demand (never loads the whole file's peaks into RAM at once) — this is the read-side counterpart that makes the streamed store actually useful downstream (e.g. for calibration/strain code that currently expects `braggvectors.get_vectors(ry, rx)`-shaped output).

- [ ] **Step 1: Write the failing test**

`tests/test_bragg_stream_reader.py`:
```python
import numpy as np

from bragg_stream import StreamingBraggWriter, read_streamed_peaks


def test_read_streamed_peaks_returns_one_positions_arrays(tmp_path):
    path = tmp_path / "stream.h5"
    with StreamingBraggWriter(path, r_shape=(1, 2)) as w:
        w.write(0, 0, qx=np.array([1.0]), qy=np.array([2.0]), intensity=np.array([9.0]))
        w.write(0, 1, qx=np.array([]), qy=np.array([]), intensity=np.array([]))

    peaks = read_streamed_peaks(path, 0, 0)
    assert peaks["qx"].tolist() == [1.0]
    assert peaks["qy"].tolist() == [2.0]
    assert peaks["intensity"].tolist() == [9.0]

    empty = read_streamed_peaks(path, 0, 1)
    assert empty["qx"].tolist() == []


def test_read_streamed_peaks_missing_position_returns_empty(tmp_path):
    path = tmp_path / "stream.h5"
    with StreamingBraggWriter(path, r_shape=(1, 1)) as w:
        w.write(0, 0, qx=np.array([1.0]), qy=np.array([1.0]), intensity=np.array([1.0]))

    # (0, 5) was never written (out of the declared r_shape, or simply skipped)
    result = read_streamed_peaks(path, 0, 5)
    assert result["qx"].tolist() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_bragg_stream_reader.py -v`
Expected: FAIL — `read_streamed_peaks` does not exist.

- [ ] **Step 3: Implement the reader**

Append to `bragg_stream.py`:
```python
def read_streamed_peaks(path: str | Path, ry: int, rx: int) -> dict:
    """Read one scan position's peaks from a streamed file — O(1) HDF5 lookup,
    never loads other positions' peaks into RAM."""
    with h5py.File(path, "r") as f:
        key = f"{ry}_{rx}"
        if key not in f["peaks"]:
            return {"qx": np.array([]), "qy": np.array([]), "intensity": np.array([])}
        arr = f["peaks"][key][()]
        return {"qx": arr[:, 0], "qy": arr[:, 1], "intensity": arr[:, 2]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_bragg_stream_reader.py -v`
Expected: 2 passed.

- [ ] **Step 5: Manual validation**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile bragg_stream.py`
Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add bragg_stream.py tests/test_bragg_stream_reader.py
git commit -m "feat: add read_streamed_peaks for on-demand per-position peak reads"
```

---

### Task 4: Real-scan validation against a full pipeline run (not automated — matches this project's own convention)

**Files:** none modified — validation only, following the same manual-real-data discipline `CHANGELOG_MIGRATION.md` Section 8 already uses for `pipeline.py` changes.

- [ ] **Step 1: Pick one real, already-validated `.mib` scan** (e.g. `Scan01_512` per `DEMO_SCRIPT.md`) with its existing `braggpeaks.h5` from a normal `compute_braggpeaks_step` run.

- [ ] **Step 2: Run `detect_braggpeaks_streaming` on the same raw datacube with identical `detect_params`**, writing to a separate output path.

- [ ] **Step 3: Compare peak counts and positions per scan position** between the existing `braggpeaks.h5` (via `py4DSTEM.read`) and the new streamed file (via `read_streamed_peaks`), for at least a 10% random sample of scan positions. Expect exact or near-exact (floating-point tolerance) agreement — any systematic mismatch means the batched `find_Bragg_disks(data=(rys, rxs), ...)` call is not equivalent to the full-scan call, and this plan's core assumption needs revisiting before it's wired into any user-facing step.

- [ ] **Step 4: Record the result** (pass/fail + peak-count deltas) as a short note appended to `MEMORY_ARCHITECTURE_REPORT.md`'s roadmap entry for this item, so the next reader knows whether the streaming path is validated on real data yet.

---

## Self-Review

**Spec coverage:** Covers the report's medium-term roadmap item 1 in full (streaming Bragg detection, reusing the existing per-position public API rather than reaching into py4DSTEM/emdfile internals) plus item 4's spirit (avoiding a full-array touch by batching, though this plan batches by count rather than routing through `dask.array` — a `dask`-based full-reduction fix for `get_origin`'s `np.max(datacube.data, axis=(0,1))` full reduction, `origin.py:265`, is a separate, smaller task not included here and can be folded into this plan later if needed).

**Placeholder scan:** Two explicit open-verification notes are flagged (the exact `BraggVectors` read-accessor name for 0.14.19) rather than silently guessed — this is a deliberate, honest gap, not a skipped step; Task 4 gates real-data adoption on resolving it.

**Type consistency:** `StreamingBraggWriter.write(ry, rx, qx, qy, intensity)` and `read_streamed_peaks(path, ry, rx) -> {"qx", "qy", "intensity"}` use matching field names throughout Tasks 1-3.
