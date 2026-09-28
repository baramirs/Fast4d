"""figsave -- save matplotlib figures without racing the GUI.

``Figure.savefig(dpi=N)`` temporarily sets ``fig.dpi = N`` on the SAME Figure
object the GUI may be showing. If a Qt canvas resizes or redraws at that moment
it sizes the figure for dpi N, and after the dpi is restored the figure stays
shrunk (at dpi=300: content in ~1/3 of the window, huge fonts). That is the
"third file" strain window in a multi-file Compute, where the worker thread
saves each file's figures while the GUI displays the previous ones.

Rule: only the main (GUI) thread calls savefig on shared figures.
``savefig`` saves at once on the main thread; on any other thread it queues the
save, and the GUI thread writes the queue with ``flush()`` (after each file,
after the batch, after any background task). Headless scripts run on the main
thread, so nothing changes for them.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

_lock = threading.Lock()
_pending: list[tuple[Any, str, dict]] = []


def _on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def savefig(fig, path, **kwargs) -> bool:
    """Save now on the main thread (True) or queue it for ``flush`` (False)."""
    if _on_main_thread():
        fig.savefig(path, **kwargs)
        return True
    with _lock:
        _pending.append((fig, str(path), dict(kwargs)))
    return False


def pending_count() -> int:
    with _lock:
        return len(_pending)


def flush(log: Callable[[str], None] | None = None) -> int:
    """Write every queued figure (main thread only). Returns how many were written;
    a failed save is logged and skipped so one bad path cannot block the rest."""
    if not _on_main_thread():
        return 0
    with _lock:
        # the same PNG queued twice (re-save, re-spill) is written once, last wins
        jobs = list({path: (fig, path, kw) for fig, path, kw in _pending}.values())
        _pending.clear()
    n = 0
    for fig, path, kwargs in jobs:
        try:
            fig.savefig(path, **kwargs)
            n += 1
        except Exception as exc:
            if log is not None:
                log(f"figure save failed ({Path(path).name}): {exc}")
    if n and log is not None:
        log(f"Wrote {n} figure PNG(s) on the GUI thread.")
    return n
