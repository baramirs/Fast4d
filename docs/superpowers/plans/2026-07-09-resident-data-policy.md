# Unified ResidentDataPolicy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the ad hoc "keep the 2 most-recent scans resident" window added in `2026-07-09-quick-wins-memory-release.md` Task 2 into a proper, configurable, independently-testable policy object — the same pattern `FigurePolicy` (`engine.py:975-1178`) already proves out for figures, applied to datacube/BVM/probe residency instead.

**Architecture:** A new sibling dataclass, `ResidentDataPolicy`, alongside the existing `FigurePolicy` — **not** a merge of the two into one god-object, to keep the blast radius on existing `get_figure_policy()`/`set_figure_policy()` call sites at zero. `qt_main.py`'s scan-switch handler is refactored to delegate its eviction decision to this policy instead of a hardcoded `[:2]` window.

**Tech Stack:** Python 3.10+, PySide6 (one new, minimal settings entry point via `QInputDialog`).

## Global Constraints

- **Prerequisite:** `2026-07-09-quick-wins-memory-release.md` Task 2 must be merged first — this plan refactors the `self._recent_scan_indices` tracking and `E.release_scans(...)` call it introduces in `qt_main.py:_on_file_selected`, rather than reintroducing them.
- Do not rename or touch `FigurePolicy`/`get_figure_policy`/`set_figure_policy` — they stay exactly as they are; this plan only adds a new, separate policy object following the same shape.
- Tests run inside `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe`.

---

### Task 1: `ResidentDataPolicy` dataclass + pure eviction-decision function

**Files:**
- Modify: `engine.py` (insert near `FigurePolicy`, e.g. directly after `_enforce_figure_ram_limit`, `engine.py:1162-1178`)
- Test: `tests/test_resident_data_policy.py`

**Interfaces:**
- Produces: `ResidentDataPolicy(max_scans_in_ram: int = 2)`; `get_data_policy() -> ResidentDataPolicy`; `set_data_policy(*, max_scans_in_ram: int | None = None) -> None`; `enforce_resident_data_limit(scans: list, active_index: int, recent_indices: list[int], *, log=None) -> list[int]` — pure function (no Qt), returns the **updated** `recent_indices` window and, as a side effect, calls `release_scans(...)` (from the prerequisite plan) on every scan that falls outside it.
- Consumed by: Task 2 (`qt_main.py` wiring).

- [ ] **Step 1: Write the failing test**

`tests/test_resident_data_policy.py`:
```python
from types import SimpleNamespace

import engine as E


def _fake_scan(datacube=None):
    return SimpleNamespace(state=SimpleNamespace(
        datacube=datacube, visualcube=None, vacuumcube=None, bvm_raw=None,
        bvm_centered=None, dp_mean=None, dp_max=None, strainmap_full=None,
        selected_disks=None, probe=None,
    ))


def test_default_policy_keeps_two_most_recent():
    assert E.get_data_policy().max_scans_in_ram == 2


def test_set_data_policy_updates_the_shared_instance():
    E.set_data_policy(max_scans_in_ram=3)
    try:
        assert E.get_data_policy().max_scans_in_ram == 3
    finally:
        E.set_data_policy(max_scans_in_ram=2)  # restore default for other tests


def test_enforce_resident_data_limit_keeps_active_plus_window_releases_rest():
    E.set_data_policy(max_scans_in_ram=2)
    scans = [_fake_scan(datacube=f"cube_{i}") for i in range(4)]

    recent = E.enforce_resident_data_limit(scans, active_index=0, recent_indices=[])
    assert recent == [0]
    assert scans[0].state.datacube == "cube_0"  # active scan untouched

    recent = E.enforce_resident_data_limit(scans, active_index=2, recent_indices=recent)
    assert recent == [2, 0]
    assert scans[2].state.datacube == "cube_2"       # newly active, untouched
    assert scans[0].state.datacube == "cube_0"        # still in the 2-item window
    assert scans[1].state.datacube is None            # released — never in the window
    assert scans[3].state.datacube is None            # released — never in the window

    recent = E.enforce_resident_data_limit(scans, active_index=1, recent_indices=recent)
    assert recent == [1, 2]
    assert scans[0].state.datacube is None  # fell out of the window this time


def test_enforce_resident_data_limit_respects_a_larger_configured_window():
    E.set_data_policy(max_scans_in_ram=3)
    try:
        scans = [_fake_scan(datacube=f"cube_{i}") for i in range(4)]
        recent = E.enforce_resident_data_limit(scans, active_index=0, recent_indices=[])
        recent = E.enforce_resident_data_limit(scans, active_index=1, recent_indices=recent)
        recent = E.enforce_resident_data_limit(scans, active_index=2, recent_indices=recent)
        assert recent == [2, 1, 0]
        assert all(scans[i].state.datacube == f"cube_{i}" for i in range(3))
    finally:
        E.set_data_policy(max_scans_in_ram=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_resident_data_policy.py -v`
Expected: FAIL — `ResidentDataPolicy`/`get_data_policy`/`set_data_policy`/`enforce_resident_data_limit` don't exist yet.

- [ ] **Step 3: Implement the policy**

Modify `engine.py` — insert right after `_enforce_figure_ram_limit` (`engine.py:1162-1178`):
```python
@dataclass
class ResidentDataPolicy:
    """How many scans' heavy compute buffers (datacube / BVM / probe) may stay
    resident in RAM at once — the same LRU-with-eviction idea FigurePolicy
    already proves out for figures (engine.py:975-1178), applied to the bigger
    objects instead. Eviction here calls the cheap `release_scans` (nulls
    references only, no gc.collect/OS trim — see its own docstring for why),
    not the heavier `free_memory`."""
    max_scans_in_ram: int = 2


_data_policy = ResidentDataPolicy()


def get_data_policy() -> ResidentDataPolicy:
    return _data_policy


def set_data_policy(*, max_scans_in_ram: int | None = None) -> None:
    """Update the global resident-data policy (GUI settings)."""
    global _data_policy
    if max_scans_in_ram is not None:
        _data_policy.max_scans_in_ram = max(1, int(max_scans_in_ram))


def enforce_resident_data_limit(scans: list, active_index: int, recent_indices: list[int],
                                *, log: Log = None) -> list[int]:
    """Update the LRU window to include ``active_index`` first, release every
    scan that falls outside the resulting window (per ``get_data_policy()``),
    and return the new window for the caller to store.

    Pure w.r.t. its inputs aside from the ``release_scans`` side effect — safe
    to call on every scan-switch."""
    limit = get_data_policy().max_scans_in_ram
    new_recent = ([active_index] + [i for i in recent_indices if i != active_index])[:limit]
    to_release = [sc for i, sc in enumerate(scans) if i not in new_recent]
    if to_release:
        release_scans(to_release, log=log)
    return new_recent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m pytest tests/test_resident_data_policy.py -v`
Expected: 4 passed.

- [ ] **Step 5: Manual validation**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile engine.py`
Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add engine.py tests/test_resident_data_policy.py
git commit -m "feat: add ResidentDataPolicy — configurable LRU window for scan residency"
```

---

### Task 2: Wire the policy into the scan-switch handler, replacing the hardcoded window

**Files:**
- Modify: `qt_main.py:5102-5112` (`_on_file_selected`, as changed by the prerequisite plan)
- Test: manual validation only (this task moves logic already covered by Task 1's unit tests into the GUI; a GUI-level regression test would duplicate Task 1's coverage without adding confidence)

**Interfaces:**
- Consumes: `engine.enforce_resident_data_limit(scans, active_index, recent_indices, *, log=None) -> list[int]` (Task 1).

- [ ] **Step 1: Replace the hardcoded window with a call to the policy**

Modify `qt_main.py:5102-5112`, changing (the version left by the prerequisite plan):
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
```
to:
```python
    def _on_file_selected(self, row: int) -> None:
        if 0 <= row < len(self._scans):
            self._active = row
            # Delegate the "how many scans stay resident" decision to the
            # configurable ResidentDataPolicy (Settings → Memory) instead of a
            # hardcoded window.
            self._recent_scan_indices = E.enforce_resident_data_limit(
                self._scans, row, self._recent_scan_indices, log=self._console.log)
```
(the rest of the method — `_update_active_views()` and the `_step` check — is unchanged.)

- [ ] **Step 2: Manual validation**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile qt_main.py`
Expected: no output, exit code 0.

Then, with 4+ scans loaded: switch between them in various orders and confirm the same behavior Task 2 of the prerequisite plan validated (instant switching, no unnecessary reload of the active scan's light ADF preview).

- [ ] **Step 3: Commit**

```bash
git add qt_main.py
git commit -m "refactor: scan-switch delegates residency window to ResidentDataPolicy"
```

---

### Task 3: Expose `max_scans_in_ram` as a user setting

**Files:**
- Modify: `qt_main.py` (add one menu action; exact insertion point depends on where the existing Settings/View menu is built — insert alongside whatever menu currently hosts `FigureStoreDialog`'s trigger, since that's the established location for memory/figure-policy settings in this GUI)
- Test: manual validation only (a `QInputDialog` round-trip is not meaningfully unit-testable without a full Qt event-loop integration test, which is out of scope for this plan)

**Interfaces:**
- Consumes: `engine.get_data_policy()` / `engine.set_data_policy(max_scans_in_ram=...)` (Task 1).

- [ ] **Step 1: Add a settings action**

Add a new method to `Fast4DWindow` (near wherever the existing figure-policy settings action is wired — find it via `grep -n "FigureStoreDialog" qt_main.py` and place this alongside it):
```python
    def _configure_resident_data_policy(self) -> None:
        """Settings → Memory → Max scans kept in RAM."""
        current = E.get_data_policy().max_scans_in_ram
        n, ok = QtWidgets.QInputDialog.getInt(
            self, "Memory settings",
            "Max scans kept fully resident in RAM\n"
            "(others release their datacube/BVM/probe on scan-switch;\n"
            "figures, ADF previews, and braggpeaks are unaffected):",
            current, 1, 20, 1)
        if ok:
            E.set_data_policy(max_scans_in_ram=n)
            self._console.log(f"Resident-data policy: max_scans_in_ram={n}")
```
Wire it to a menu action the same way the existing `FigureStoreDialog` trigger is wired (same menu, one entry below/above it) — e.g. `act.triggered.connect(self._configure_resident_data_policy)`.

- [ ] **Step 2: Manual validation**

Run: `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe -m py_compile qt_main.py`
Expected: no output, exit code 0.

Then in the running app: open the new menu action, set it to `1`, load 3 scans, switch between all 3, and confirm only the currently-active scan's heavy buffers stay resident (i.e. switching away from any scan immediately releases it) — then set it back to `2` (default) and confirm the previous, gentler behavior returns.

- [ ] **Step 3: Commit**

```bash
git add qt_main.py
git commit -m "feat: expose max_scans_in_ram as a user-configurable memory setting"
```

---

## Self-Review

**Spec coverage:** Covers the report's medium-term roadmap item 2 in full ("Generalize FigurePolicy... into a single ResidentDataPolicy... instead of today's per-field ad hoc handling") — implemented as a sibling policy rather than a literal merge, a deliberate, lower-risk refinement of the report's wording, explained in the Architecture section.

**Placeholder scan:** Task 3's menu-wiring step references "wherever the existing figure-policy settings action is wired" rather than a fabricated exact line number, because this plan's investigation did not read `FigureStoreDialog`'s full call site — this is flagged as a find-it-yourself step (via a one-line `grep`) rather than a guessed citation, consistent with not inventing unverified specifics.

**Type consistency:** `enforce_resident_data_limit(scans, active_index, recent_indices, *, log=None) -> list[int]` signature matches between Task 1's test, its implementation, and Task 2's call site.
