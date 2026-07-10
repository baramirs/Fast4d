# Quick Wins — Automatic Memory Release & Safer Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Fast4D's existing, already-correct memory-release machinery (`engine.free_memory`, `FigurePolicy`) fire automatically at the moments the codebase itself already identifies as expensive, instead of only from a manual "Free RAM" button — with zero changes to scientific/numeric behavior.

**Architecture:** No new subsystem. Each task is a small, additive change at a specific call site already identified in `MEMORY_ARCHITECTURE_REPORT.md` Part I/Deliverable 2. A new lightweight `engine.release_scans()` helper (cheap: nulls references, no `gc.collect()`/OS trim) complements the existing, heavier `engine.free_memory()` (kept as-is, just called more often).

**Tech Stack:** Python 3.10+, PySide6, py4DSTEM==0.14.19 (all already pinned in `requirements.txt`); `pytest` added as a new dev-only dependency (none exists in the repo today).

## Global Constraints

- No test infrastructure exists in this repo today (confirmed: no `tests/` dir, no `pytest` in `requirements.txt`, no `conftest.py`). This plan's Task 1 creates it.
- `engine.py`/`pipeline.py`/`qt_main.py`/`driver.py` import `py4DSTEM`/`PySide6` at module level, so tests that `import` them must run inside the pinned environment: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe`.
- Do not change any numeric/scientific result — this plan is memory-lifecycle-only, matching the discipline already documented in `CHANGELOG_MIGRATION.md` ("no se modificaron algoritmos científicos, resultados numéricos").
- Every task must leave the app important behaviors identical for a user who never runs a "Free RAM" click today — these changes only affect *when* memory is released, never *what* the GUI can compute or display.
- Follow existing code style: `_log(log, ...)` for logging, `try/except Exception: pass` for best-effort cleanup (matching `engine.free_memory`'s own style at `engine.py:2903-2974`).

---

### Task 1: Test infrastructure + safer default for large-file memory mode

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_mem_mode.py`
- Create: `requirements-dev.txt`
- Modify: `pipeline.py:328-353` (`_pick_mib_mem_mode`)

**Interfaces:**
- Produces: `pipeline._pick_mib_mem_mode(path) -> str | None` (unchanged signature, new thresholds + `FAST4D_FORCE_MEMMAP` env-var override). No other task depends on this one — it's the first task purely to bootstrap `tests/` for everyone after it.

- [ ] **Step 1: Create the dev requirements file**

`requirements-dev.txt`:
```
-r requirements.txt
pytest>=7.0
```

- [ ] **Step 2: Install it into the pinned environment**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pip install -r requirements-dev.txt`
Expected: `pytest` installs without touching the pinned `py4DSTEM==0.14.19`.

- [ ] **Step 3: Create the shared test config**

`tests/conftest.py`:
```python
import sys
from pathlib import Path

# Fast4D has no package layout / setup.py — tests import top-level modules
# (pipeline.py, engine.py, ...) directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(scope="session")
def qapp():
    """Shared QApplication for tests that construct Qt widgets off-screen."""
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
```

- [ ] **Step 4: Write the failing test for the new heuristic**

`tests/test_mem_mode.py`:
```python
import pipeline


class _FakeStat:
    def __init__(self, size):
        self.st_size = size


class _FakePath:
    """Duck-types the one method _pick_mib_mem_mode actually calls."""
    def __init__(self, size):
        self._size = size

    def stat(self):
        return _FakeStat(self._size)


def test_force_memmap_env_var_wins(monkeypatch):
    monkeypatch.setenv("FAST4D_FORCE_MEMMAP", "1")
    assert pipeline._pick_mib_mem_mode(_FakePath(1024)) == "memmap"


def test_small_file_under_thresholds_returns_none(monkeypatch):
    monkeypatch.delenv("FAST4D_FORCE_MEMMAP", raising=False)
    assert pipeline._pick_mib_mem_mode(_FakePath(10 * 1024**2)) is None


def test_file_at_2gib_returns_memmap(monkeypatch):
    monkeypatch.delenv("FAST4D_FORCE_MEMMAP", raising=False)
    assert pipeline._pick_mib_mem_mode(_FakePath(2 * 1024**3)) == "memmap"
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_mem_mode.py -v`
Expected: `test_force_memmap_env_var_wins` and `test_file_at_2gib_returns_memmap` FAIL (current code has no env-var override and a 6 GiB threshold, not 2 GiB).

- [ ] **Step 6: Implement the new heuristic**

Modify `pipeline.py:328-353`, replacing the existing `_pick_mib_mem_mode`:
```python
def _pick_mib_mem_mode(path: Path) -> str | None:
    """
    Choose a safe default for `read_mib.load_mib(mem=...)`.

    Large 4D cubes (e.g. 512x512x256x256) are ~32GB as float32; forcing RAM is often
    impossible. An explicit FAST4D_FORCE_MEMMAP=1 environment variable always wins,
    for memory-constrained machines or users who know their datasets are large.
    """
    import os

    if os.environ.get("FAST4D_FORCE_MEMMAP", "").strip() == "1":
        return "memmap"

    try:
        size = int(path.stat().st_size)
    except Exception:
        size = 0

    # Heuristic: if the file is at least moderately large, prefer memmap even before
    # we know the final shape. Lowered from 6 GiB to 2 GiB (2026-07-09 memory report):
    # most workstation RAM budgets can't afford more than a couple of RAM-resident
    # cubes at 2+ GiB anyway, and memmap's cost is amortized I/O, not correctness risk.
    if size >= 2 * 1024**3:  # >= ~2 GiB on disk
        return "memmap"

    try:
        import psutil  # type: ignore

        avail = int(psutil.virtual_memory().available)
        # If the file is > ~25% of available RAM, don't default to RAM (was 40%).
        if size > 0 and size > int(0.25 * max(avail, 1)):
            return "memmap"
    except Exception:
        pass

    return None
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_mem_mode.py -v`
Expected: 3 passed.

- [ ] **Step 8: Manual validation (matches this project's existing convention, no automated GPU/file test)**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile pipeline.py`
Expected: no output, exit code 0.

- [ ] **Step 9: Commit**

```bash
git add tests/conftest.py tests/test_mem_mode.py requirements-dev.txt pipeline.py
git commit -m "feat: default to memmap sooner + add FAST4D_FORCE_MEMMAP override"
```

---

### Task 2: Cheap automatic release of inactive scans on scan-switch

**Files:**
- Modify: `engine.py:2899-2903` (insert new helpers just before `free_memory`)
- Modify: `qt_main.py:3879` (init) and `qt_main.py:5102-5112` (`_on_file_selected`)
- Test: `tests/test_release_scans.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `engine.release_scans(scans: list, *, log=None) -> int` and `engine._close_memmap_handle(state) -> None` — Task 3 does **not** use these (it uses `free_memory` directly), but a later "ResidentDataPolicy" plan builds on `release_scans` and on `qt_main.py`'s new `self._recent_scan_indices` list.

- [ ] **Step 1: Write the failing test**

`tests/test_release_scans.py`:
```python
from types import SimpleNamespace

import engine as E


def _fake_scan(**heavy_attrs):
    defaults = dict(datacube=None, visualcube=None, vacuumcube=None, bvm_raw=None,
                     bvm_centered=None, dp_mean=None, dp_max=None, strainmap_full=None,
                     selected_disks=None, probe=None)
    defaults.update(heavy_attrs)
    return SimpleNamespace(state=SimpleNamespace(**defaults))


def test_release_scans_nulls_heavy_attrs_and_counts_them():
    sc = _fake_scan(datacube="fake_cube", probe="fake_probe")
    n = E.release_scans([sc])
    assert n == 2
    assert sc.state.datacube is None
    assert sc.state.probe is None


def test_release_scans_skips_scans_with_no_state():
    sc = SimpleNamespace(state=None)
    assert E.release_scans([sc]) == 0


def test_release_scans_handles_empty_or_none_input():
    assert E.release_scans([]) == 0
    assert E.release_scans(None) == 0


def test_release_scans_does_not_touch_figures_or_braggpeaks():
    sc = _fake_scan(datacube="fake_cube")
    sc.state.figures = {"origin": "fake_fig"}
    sc.state.braggpeaks = "fake_peaks"
    E.release_scans([sc])
    assert sc.state.figures == {"origin": "fake_fig"}
    assert sc.state.braggpeaks == "fake_peaks"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_release_scans.py -v`
Expected: FAIL with `AttributeError: module 'engine' has no attribute 'release_scans'`.

- [ ] **Step 3: Implement `release_scans` + shared memmap-close helper**

Modify `engine.py` — insert immediately before the existing `def free_memory(...)` at `engine.py:2903` (i.e. right after the `# PERSIST` section header at line 2901-2902):
```python
def _close_memmap_handle(st) -> None:
    """Close a memmap-backed datacube's file handle before the ref is dropped,
    else the OS keeps the mapping resident. Shared by ``free_memory`` and the
    cheaper ``release_scans``."""
    dc = getattr(st, "datacube", None)
    try:
        data = getattr(dc, "data", None)
        base = getattr(data, "base", None)
        for obj in (data, base):
            mm = getattr(obj, "_mmap", None) or (obj if "mmap" in type(obj).__name__.lower() else None)
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
    except Exception:
        pass


def release_scans(scans: list, *, log: Log = None) -> int:
    """Cheap, synchronous variant of :func:`free_memory` for the scan-switch path.

    Closes memmap handles and nulls the same heavy attributes as ``free_memory``
    (datacube / visualcube / vacuumcube / BVM histograms / probe), but skips the
    3x ``gc.collect()``, OS working-set trim, and CuPy pool free — those cost tens
    to hundreds of ms on a large heap, and the scan-switch handler must stay cheap
    (``qt_main.py`` ``_on_file_selected``: "Selection must stay CHEAP"). Figures and
    braggpeaks are intentionally left alone — those are governed by ``FigurePolicy``
    and the explicit "Free RAM" button respectively.

    Call :func:`free_memory` (already wired to "Free RAM" and to batch completion)
    to reclaim the actual OS/GPU memory once several scans have accumulated
    None-ed buffers.
    """
    n = 0
    for sc in (scans or []):
        st = getattr(sc, "state", None)
        if st is None:
            continue
        _close_memmap_handle(st)
        for a in ("datacube", "visualcube", "vacuumcube", "bvm_raw", "bvm_centered",
                  "dp_mean", "dp_max", "strainmap_full", "selected_disks", "probe"):
            if getattr(st, a, None) is not None:
                try:
                    setattr(st, a, None); n += 1
                except Exception:
                    pass
    if n:
        _log(log, f"Released {n} heavy buffer(s) from {len(scans)} inactive scan(s) (cheap pass).")
    return n
```

Also modify the existing `free_memory` (`engine.py:2918-2932`) to reuse the new helper instead of duplicating the memmap-close block — replace those 15 lines with:
```python
        _close_memmap_handle(st)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_release_scans.py -v`
Expected: 4 passed.

- [ ] **Step 5: Wire it into the scan-switch handler**

Modify `qt_main.py:3879` — add a new tracking list right after `self._active = -1`:
```python
        self._active = -1
        self._recent_scan_indices: list[int] = []  # LRU window for release_scans()
```

Modify `qt_main.py:5102-5112` (`_on_file_selected`):
```python
    def _on_file_selected(self, row: int) -> None:
        if 0 <= row < len(self._scans):
            self._active = row
            # Keep the 2 most-recently-viewed scans' heavy buffers resident (active +
            # one back-reference for quick A/B comparison); release the rest. This is
            # a cheap pass (no gc.collect/OS trim) — see engine.release_scans docstring.
            self._recent_scan_indices = ([row] + [i for i in self._recent_scan_indices if i != row])[:2]
            to_release = [sc for i, sc in enumerate(self._scans) if i not in self._recent_scan_indices]
            if to_release:
                E.release_scans(to_release, log=self._console.log)
            # Selection must stay CHEAP: just show the cached/.h5 ADF preview.
            # braggpeaks.h5 is NOT loaded here — that py4DSTEM read is slow and the
            # user only wants to see the ADF. It loads lazily when a calibration
            # tool is opened (_setting_calibration / _apply_calib_step / dialogs
            # all call ensure_braggpeaks_for_calibration themselves).
            self._update_active_views()
            if getattr(self, "_step", None) in ("probe", "select6", "detection"):
                self._populate_step_actions(self._step)
```

- [ ] **Step 6: Manual validation**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile engine.py qt_main.py`
Expected: no output, exit code 0.

Then, with the app running against at least 3 loaded scans (per the existing manual-QA convention in `CHANGELOG_MIGRATION.md`): click between scan A → B → C → A in the Files list; confirm each switch stays instant (no visible hitch) and that re-selecting scan A does not require reloading its `.mib` from disk (only its previously-computed products need recomputation if they were released — the light ADF preview must still show immediately).

- [ ] **Step 7: Commit**

```bash
git add engine.py qt_main.py tests/test_release_scans.py
git commit -m "feat: release inactive scans' heavy buffers on scan-switch (cheap pass)"
```

---

### Task 3: Automatic release after each scan in the batch driver

**Files:**
- Modify: `driver.py:359-364` (`compute_all`)
- Test: `tests/test_driver_release.py`

**Interfaces:**
- Consumes: `engine.free_memory` (existing, unchanged signature: `free_memory(scans: list, *, drop_braggpeaks=False, log=None) -> dict`).
- Produces: nothing new consumed by later tasks.

- [ ] **Step 1: Write the failing test**

`tests/test_driver_release.py`:
```python
from types import SimpleNamespace

import driver
import engine as E


def test_compute_all_releases_memory_after_each_scan(monkeypatch):
    scans = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    fake_outcome = SimpleNamespace(ok=True, error=None, elapsed_s=0.1)
    monkeypatch.setattr(driver, "compute_scan", lambda *a, **k: fake_outcome)

    release_calls = []
    monkeypatch.setattr(E, "free_memory", lambda scans_arg, **k: release_calls.append(list(scans_arg)))

    driver.compute_all(scans)

    assert release_calls == [[scans[0]], [scans[1]]]


def test_compute_all_still_calls_on_scan_done_before_releasing(monkeypatch):
    scans = [SimpleNamespace(name="a")]
    fake_outcome = SimpleNamespace(ok=True, error=None, elapsed_s=0.1)
    monkeypatch.setattr(driver, "compute_scan", lambda *a, **k: fake_outcome)
    order = []
    monkeypatch.setattr(E, "free_memory", lambda *a, **k: order.append("release"))

    driver.compute_all(scans, on_scan_done=lambda out: order.append("on_scan_done"))

    assert order == ["on_scan_done", "release"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_driver_release.py -v`
Expected: FAIL — `release_calls` stays empty (no release happens today).

- [ ] **Step 3: Implement the release call**

Modify `driver.py:359-364` (inside `compute_all`'s loop), changing:
```python
        if on_scan_done is not None:
            try:
                on_scan_done(out)
            except Exception:
                pass
        if out.error == "cancelled" or (cancel is not None and cancel()):
```
to:
```python
        if on_scan_done is not None:
            try:
                on_scan_done(out)
            except Exception:
                pass
        # Release this scan's heavy buffers now that it's saved/reported — mirrors
        # the manual "Free RAM" button (qt_main.py:4764-4775). Figures, ADF cache,
        # and braggpeaks survive (drop_braggpeaks=False), so on_scan_done's GUI
        # update above already had everything it needs before this line runs.
        E.free_memory([scan], log=log)
        if out.error == "cancelled" or (cancel is not None and cancel()):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_driver_release.py -v`
Expected: 2 passed.

- [ ] **Step 5: Manual validation**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile driver.py`
Expected: no output, exit code 0.

Then run a real multi-scan Compute batch (3+ `.mib` files) via the GUI's batch compute path and watch Task Manager / `nvidia-smi`: resident RAM should no longer grow roughly linearly with scan count; it should plateau after the first scan or two.

- [ ] **Step 6: Commit**

```bash
git add driver.py tests/test_driver_release.py
git commit -m "feat: auto-release each scan's heavy buffers after driver.compute_all finishes it"
```

---

### Task 4: `gc.collect()` right after the datacube+braggpeaks double-residency point

**Files:**
- Modify: `pipeline.py:1420-1433` (`compute_braggpeaks_step`)
- Test: `tests/test_gc_after_save.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

`tests/test_gc_after_save.py`:
```python
from types import SimpleNamespace

import pipeline


def test_compute_braggpeaks_step_collects_garbage_after_save(monkeypatch, tmp_path):
    fake_peaks = object()
    fake_datacube = SimpleNamespace(find_Bragg_disks=lambda **kw: fake_peaks)
    state = SimpleNamespace(
        use_existing_braggpeaks=False, braggpeaks=None, datacube=fake_datacube,
        probe=object(), detect_params={}, braggpeaks_path=None,
    )
    monkeypatch.setattr(pipeline, "normalize_subpixel_keyword", lambda x: x)
    monkeypatch.setattr(pipeline, "cupy_device_summary", lambda: "")
    monkeypatch.setattr(pipeline, "_ensure_cupy_current_device_for_thread", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "format_probe_template_log_line", lambda st: "")
    monkeypatch.setattr(pipeline, "probe_kernel_template_ndarray", lambda p: object())
    monkeypatch.setattr(pipeline, "save_braggpeaks_datacube_notebook_style", lambda *a, **k: None)

    gc_calls = []
    monkeypatch.setattr("gc.collect", lambda: gc_calls.append(1))

    pipeline.compute_braggpeaks_step(state, save_path=tmp_path / "bp.h5")

    assert gc_calls, "gc.collect() should run right after the notebook-style save completes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_gc_after_save.py -v`
Expected: FAIL — `gc_calls` is empty.

- [ ] **Step 3: Implement the fix**

Modify `pipeline.py:1430-1433`, changing:
```python
        save_braggpeaks_file(state.braggpeaks, out_path, log=log)
    state.braggpeaks_path = out_path
    _log(log, f"Full-scan braggpeaks ready: {out_path}")
    return state.braggpeaks
```
to:
```python
        save_braggpeaks_file(state.braggpeaks, out_path, log=log)
    state.braggpeaks_path = out_path
    _log(log, f"Full-scan braggpeaks ready: {out_path}")
    # The full datacube and the freshly detected braggpeaks were both alive at once
    # during the save above (py4DSTEM.save(path, dc, ...) — pipeline.py:1471); this
    # is the one designed-in double-residency point in the pipeline. A single
    # gc.collect() here reclaims any transient copies py4DSTEM's own save path made
    # (e.g. internal serialization buffers) before the next step runs.
    import gc
    gc.collect()
    return state.braggpeaks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_gc_after_save.py -v`
Expected: 1 passed.

- [ ] **Step 5: Manual validation**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile pipeline.py`
Expected: no output, exit code 0. Then run a real full-scan Bragg detection + save on one `.mib` and confirm the saved `braggpeaks.h5` is byte-identical in structure to a pre-change run (open both with `py4DSTEM.read` and compare `.braggpeaks` peak counts) — this change must not alter what gets saved, only add a GC pass after.

- [ ] **Step 6: Commit**

```bash
git add pipeline.py tests/test_gc_after_save.py
git commit -m "fix: gc.collect() after the datacube+braggpeaks double-residency save point"
```

---

### Task 5: Remove unwired `BatchScanResult`/`BatchScanItem` scaffolding

**Files:**
- Modify: `batch_common.py:176-543` (delete)
- Test: manual validation only (deleting confirmed-dead code has no new behavior to unit-test)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (this is a deletion).

**Decision (per YAGNI and the report's Deliverable 2, bottleneck #7):** `BatchScanItem`, `BatchScanResult`, `extract_scan_result`, `batch_scan_cal_flags`, and `batch_item_cal_ui_flags` (`batch_common.py:176-543`) have **zero references anywhere else in the repository** — confirmed by `grep -rn "BatchScanResult\|BatchScanItem\|extract_scan_result\|batch_scan_cal_flags\|batch_item_cal_ui_flags"` outside `batch_common.py` returning nothing. This is unwired scaffolding for a batch-plugin feature that was never connected to `qt_main.py`. Per the report: "either wire it with the same LRU+spill discipline as `FigurePolicy`, or remove it." Wiring it now would be speculative (no current caller, no confirmed feature spec) — YAGNI says remove it. If the batch-plugin feature is revived later, it should be designed against the (by-then) `ResidentDataPolicy` from the start, not retrofitted onto dead code.

- [ ] **Step 1: Confirm the boundary and zero-reference claim are still true**

Run: `grep -n "^class \|^def \|^@dataclass" batch_common.py`
Expected: `BatchScanItem` at 176, `BatchScanResult` at 206, `extract_scan_result` at 280, `batch_scan_cal_flags` at 402, `batch_item_cal_ui_flags` at 423, then `batch_calibration_display_name` at 544 (first *live* function after the dead block).

Run: `grep -rn "BatchScanResult\|BatchScanItem\|extract_scan_result\|batch_scan_cal_flags\|batch_item_cal_ui_flags" . --include="*.py" | grep -v "batch_common.py"`
Expected: no output (zero external references).

- [ ] **Step 2: Delete the dead block**

Modify `batch_common.py`: delete lines 176 through 543 inclusive (from `@dataclass` above `class BatchScanItem` through the end of `batch_item_cal_ui_flags`, i.e. everything up to but not including `def batch_calibration_display_name` at line 544).

- [ ] **Step 3: Clean up now-unused imports**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -c "import ast, sys; tree = ast.parse(open('batch_common.py', encoding='utf-8').read()); print('parsed OK')"`
Then manually check the top-of-file imports (`dataclass`, `field`, `Any`, `np`) against what's still used in the remaining ~340 lines; remove any import that's now unused (only if `grep -c` for that name in the trimmed file is 1, i.e. only the import line itself remains).

- [ ] **Step 4: Manual validation**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile batch_common.py`
Expected: no output, exit code 0.

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -c "import batch_common"`
Expected: no output, exit code 0 (confirms no other module was silently depending on the removed names via `import *` or similar).

- [ ] **Step 5: Commit**

```bash
git add batch_common.py
git commit -m "refactor: remove unwired BatchScanResult/BatchScanItem scaffolding (zero callers)"
```

---

### Task 6: Lazy figure resolution for `ClickableFigureLabel` (params-table call site)

**Files:**
- Modify: `qt_widgets.py:785-818` (`ClickableFigureLabel`)
- Modify: `qt_params.py:102-103` (call site)
- Test: `tests/test_lazy_figure_label.py`

**Interfaces:**
- Consumes: `engine.resolve_figure(scan, key)` (existing, unchanged, `engine.py:1104-1112`).
- Produces: `ClickableFigureLabel(fig, *, spill_path="", title="Figure", parent=None, max_w=240, max_h=150, dpi=80, scan=None, fig_key="")` — the two new keyword-only params are optional and default to today's behavior (permanent `self._fig` cache) when omitted, so `qt_report.py:461-462`'s call site (which has no `(scan, key)` context available — its figures come from ad hoc report queries, not the `FigurePolicy`-managed `scan.figures` pool) is deliberately left unchanged.

- [ ] **Step 1: Write the failing test**

`tests/test_lazy_figure_label.py`:
```python
from types import SimpleNamespace

from matplotlib.figure import Figure

import engine as E
from qt_widgets import ClickableFigureLabel


def _tiny_figure():
    return Figure()


def test_lazy_label_follows_scan_figures_instead_of_caching(qapp):
    fig = _tiny_figure()
    scan = SimpleNamespace(figures={"origin": fig}, figure_spill={})
    label = ClickableFigureLabel(fig, title="t", scan=scan, fig_key="origin")

    assert label._fig is None  # never caches its own permanent reference in lazy mode

    # Simulate FigurePolicy eviction + a fresh recompute under the same key.
    scan.figures.pop("origin")
    E._close_figure(fig)
    fresh_fig = _tiny_figure()
    scan.figures["origin"] = fresh_fig

    assert label._resolve_fig() is fresh_fig


def test_label_without_scan_context_keeps_old_permanent_cache_behavior(qapp):
    fig = _tiny_figure()
    label = ClickableFigureLabel(fig, title="t")  # qt_report.py-style call, unchanged
    assert label._fig is fig
    assert label._resolve_fig() is fig
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_lazy_figure_label.py -v`
Expected: FAIL — `ClickableFigureLabel.__init__` doesn't accept `scan`/`fig_key` yet.

- [ ] **Step 3: Implement the lazy-resolve mode**

Modify `qt_widgets.py:785-818`, replacing the whole class with:
```python
class ClickableFigureLabel(QtWidgets.QLabel):
    """A figure thumbnail; click → open it full-size (with residuals etc.).

    Pass ``scan``/``fig_key`` so the label re-resolves the Figure lazily on
    click via ``engine.resolve_figure`` (RAM-or-spilled-PNG) instead of holding
    its own permanent reference. A permanent reference can outlive
    ``FigurePolicy``'s eviction of the same Figure from ``scan.figures``
    (engine.py:1162-1178), keeping it — and the arrays it plots — resident
    longer than the policy intends. Omit ``scan``/``fig_key`` to keep today's
    behavior (used by qt_report.py's ad hoc report figures, which aren't part
    of the ``scan.figures`` pool in the first place).
    """

    def __init__(self, fig, *, spill_path: str = "", title: str = "Figure",
                 parent=None, max_w: int = 240, max_h: int = 150, dpi: int = 80,
                 scan=None, fig_key: str = "") -> None:
        super().__init__(parent)
        self._scan = scan
        self._fig_key = fig_key
        self._spill_path = spill_path or ""
        self._title = title
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(110, 84)
        if fig is not None:
            pix = figure_to_pixmap(fig, max_w, max_h, dpi=dpi)
            if not pix.isNull():
                self.setPixmap(pix)
            self.setToolTip("Click to enlarge")
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.setStyleSheet("border:1px solid #cfd8dc;")
        elif self._spill_path:
            pix = figure_to_pixmap(None, max_w, max_h, png_path=self._spill_path)
            if not pix.isNull():
                self.setPixmap(pix)
            self.setToolTip("Click to enlarge (spilled to disk)")
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.setStyleSheet("border:1px dashed #90A4AE;")
        else:
            self.setText("—")
            self.setStyleSheet("color:#aaa;")
        # Only cache a direct Figure reference when there's no (scan, fig_key) to
        # re-resolve from later — lazy mode is preferred whenever it's available.
        self._fig = fig if (scan is None or not fig_key) else None

    def _resolve_fig(self):
        if self._scan is not None and self._fig_key:
            import engine as E
            return E.resolve_figure(self._scan, self._fig_key)
        return self._fig

    def mousePressEvent(self, ev) -> None:
        fig = self._resolve_fig()
        if fig is not None:
            FigureDialog(fig, self.window(), self._title).exec()
        elif self._spill_path:
            FigureDialog.from_png(self._spill_path, self.window(), self._title).exec()
```

- [ ] **Step 4: Wire the params-table call site**

Modify `qt_params.py:102-103`, changing:
```python
                    self.setCellWidget(row, c, ClickableFigureLabel(
                        fig, spill_path=spill, title=f"{sc.name} — {fk}"))
```
to:
```python
                    self.setCellWidget(row, c, ClickableFigureLabel(
                        fig, spill_path=spill, title=f"{sc.name} — {fk}",
                        scan=sc, fig_key=fk))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_lazy_figure_label.py -v`
Expected: 2 passed.

- [ ] **Step 6: Manual validation**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile qt_widgets.py qt_params.py`
Expected: no output, exit code 0. Then in the running app: open the parameter-table figure thumbnails for a scan, trigger enough other figure activity to push that scan past `FigurePolicy.max_in_ram` (12), and confirm clicking the now-evicted thumbnail still opens something sensible (either the freshly recomputed figure or the spilled-PNG fallback), never a stale closed Figure.

- [ ] **Step 7: Commit**

```bash
git add qt_widgets.py qt_params.py tests/test_lazy_figure_label.py
git commit -m "fix: ClickableFigureLabel resolves lazily via engine.resolve_figure (params table)"
```

---

## Self-Review

**Spec coverage:** All 6 "Quick wins" from `MEMORY_ARCHITECTURE_REPORT.md`'s roadmap are covered 1:1 by Tasks 1-6 (Task 1 = roadmap item 3 + infra bootstrap; Task 2 = item 1's scan-switch half; Task 3 = item 1's batch half / item 2; Task 4 = item 4; Task 5 = item 5; Task 6 = item 6).

**Placeholder scan:** No TBD/TODO markers; every step has runnable code and an exact expected result.

**Type consistency:** `release_scans(scans: list, *, log=None) -> int` used identically in Task 2's test and wiring. `ClickableFigureLabel(..., scan=None, fig_key="")` keyword names match between the class definition (Task 6, Step 3) and both call sites (Task 6 Step 4; `qt_report.py` explicitly left unchanged and noted why).
