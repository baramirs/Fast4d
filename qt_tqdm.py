"""Throttled tqdm → GUI bridge (fixes the window freeze during heavy loads).

py4DSTEM/cupy drive their progress with tqdm, which writes a `\r`-redrawn bar to
stdout thousands of times per second. When that stream is the cmd.exe console
(run_gui.bat launches python directly), each write is a SLOW Windows console
operation; issued from the worker thread it monopolises the GIL and starves the
Qt GUI thread → the whole window freezes mid-load.

The fix (ported from the Tk BackUp2 ``_patch_tqdm_for_gui_console``):
  * send the raw tqdm bar to a /dev/null sink — the `\r` storm never reaches the
    slow console, so the GIL is no longer hammered;
  * instead emit ONE newline-terminated progress line, THROTTLED to ~1/s (or a
    ≥2 % jump), to the registered sinks — the GUI console + the progress bar.

``install()`` MUST run before py4DSTEM imports tqdm (i.e. before the heavy warmup),
otherwise py4DSTEM keeps a reference to the original tqdm class. ``register_sinks``
is called later, once the main window (console + progress bar) exists. Both sinks
are invoked from whatever thread tqdm runs on, so they must be thread-safe (the
GUI wires them through Qt queued signals).
"""
from __future__ import annotations

import time
from typing import Callable

_progress_sinks: list[Callable[[float], None]] = []
_console_sinks: list[Callable[[str], None]] = []
_installed = False


def register_sinks(*, progress: Callable[[float], None] | None = None,
                   console: Callable[[str], None] | None = None) -> None:
    """Wire a persistent, app-lifetime sink pair (thread-safe callables) — the
    main window calls this once at startup for its own bar + console."""
    if progress is not None and progress not in _progress_sinks:
        _progress_sinks.append(progress)
    if console is not None and console not in _console_sinks:
        _console_sinks.append(console)


def add_temp_sinks(*, progress: Callable[[float], None] | None = None,
                   console: Callable[[str], None] | None = None) -> Callable[[], None]:
    """Register additional sink(s) for as long as one tool window is open (e.g.
    a dialog's own progress bar), on top of any persistent ones. Returns a
    ``remove()`` callback — call it when the window closes so events stop
    reaching a dead widget."""
    added: list[tuple[list, Callable]] = []
    if progress is not None:
        _progress_sinks.append(progress)
        added.append((_progress_sinks, progress))
    if console is not None:
        _console_sinks.append(console)
        added.append((_console_sinks, console))

    def remove() -> None:
        for lst, cb in added:
            try:
                lst.remove(cb)
            except ValueError:
                pass

    return remove


def _emit_line(text: str) -> None:
    sent = False
    for cb in list(_console_sinks):
        try:
            cb(text)
            sent = True
        except Exception:
            pass
    if sent:
        return
    # Before the GUI console exists (warmup), fall back to a plain newline print —
    # NOT the \r bar, so it never spams the slow console.
    try:
        print(text, flush=True)
    except Exception:
        pass


def _bar(pct: float, width: int = 28) -> str:
    filled = max(0, min(width, int(round(width * pct / 100.0))))
    return ("#" * filled) + ("." * (width - filled))


class _DevNull:
    """Swallow the raw tqdm bar (the \\r spam) so it never hits the real console."""

    def write(self, _s: str) -> None:
        pass

    def flush(self) -> None:
        pass


def install() -> None:
    """Replace tqdm with a throttled, GUI-friendly subclass. Idempotent."""
    global _installed
    if _installed:
        return
    try:
        import tqdm as _tqdm_mod
    except Exception:
        return

    _orig = _tqdm_mod.tqdm

    class _GuiTqdm(_orig):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            # tqdm.__del__ may run even if parent __init__ fails before creating
            # attributes like last_print_t. Pre-seed our own fields and make close()
            # tolerate partially initialized instances.
            self._c_last_t = 0.0
            self._c_last_pct = -1.0
            self._c_started = False
            self._c_closed = False
            kw = dict(kwargs)
            # Always discard the raw \r bar, even if py4DSTEM/tqdm passes
            # file=sys.stderr explicitly. We emit throttled GUI-safe lines below.
            kw["file"] = _DevNull()
            kw.setdefault("mininterval", 0.8)     # tqdm's own update throttle
            kw.setdefault("maxinterval", 3.0)
            super().__init__(*args, **kw)

        def update(self, n=1):  # type: ignore[override]
            try:
                return super().update(n)
            finally:
                self._emit()

        def refresh(self, *args, **kwargs):  # type: ignore[override]
            if not hasattr(self, "last_print_t"):
                return None
            r = super().refresh(*args, **kwargs)
            self._emit()
            return r

        def close(self, *args, **kwargs):  # type: ignore[override]
            if not hasattr(self, "last_print_t"):
                return None
            if getattr(self, "_c_closed", False):
                return None
            try:
                return super().close(*args, **kwargs)
            finally:
                self._c_closed = True
                self._emit(final=True)

        def _emit(self, final: bool = False) -> None:
            try:
                if getattr(self, "disable", False) and not final:
                    return
                desc = str(getattr(self, "desc", "") or "").strip() or "progress"
                n = int(getattr(self, "n", 0))
                tot_raw = getattr(self, "total", None)
                now = time.monotonic()

                if tot_raw is not None and float(tot_raw) > 0:
                    tot = float(tot_raw)
                    pct = 100.0 * min(max(n, 0), tot) / tot
                    finished = final or n >= int(tot)
                    if not self._c_started:
                        self._c_started = True
                        self._c_last_t = now
                        self._c_last_pct = 0.0
                    # THROTTLE: line + progress-bar update at most ~1/s or every ≥2 %.
                    if finished or (pct - self._c_last_pct) >= 2.0 or (now - self._c_last_t) >= 1.0:
                        for cb in list(_progress_sinks):
                            try:
                                cb(pct)
                            except Exception:
                                pass
                        _emit_line(
                            f"[{desc}] {pct:5.1f}%  [{_bar(pct)}] ({n}/{int(tot)})"
                            + ("  done" if finished else ""))
                        self._c_last_t = now
                        self._c_last_pct = pct
                else:
                    # Unknown total: show activity every ~0.6 s.
                    if not self._c_started:
                        self._c_started = True
                        self._c_last_t = now
                        _emit_line(f"[{desc}] start")
                    if final or (now - self._c_last_t) >= 0.6:
                        _emit_line(f"[{desc}] processed {n}" + ("  done" if final else ""))
                        self._c_last_t = now
            except Exception:
                pass

    _tqdm_mod.tqdm = _GuiTqdm  # type: ignore[assignment]
    for modname in ("tqdm.auto", "tqdm.std", "tqdm.notebook"):
        try:
            import importlib
            m = importlib.import_module(modname)
            m.tqdm = _GuiTqdm  # type: ignore[attr-defined]
        except Exception:
            pass
    _installed = True
