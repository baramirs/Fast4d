"""session_scope -- which files the Report's Session analysis acts on.

Qt-free. ``checked`` mirrors the tick boxes in Session analysis > Files;
``applied`` is what the analyses use and changes only on ``refresh()`` (the
Refresh button), so ticking boxes never silently changes what a figure shows.
One selection per session (``current()``), saved in the session JSON by engine.
"""
from __future__ import annotations


class SessionSelection:
    def __init__(self) -> None:
        self.checked: list[str] = []
        self.applied: list[str] = []

    def set_checked(self, names) -> None:
        self.checked = [str(n) for n in names]

    def refresh(self) -> None:
        self.applied = list(self.checked)

    @property
    def pending(self) -> bool:
        """Ticks differ from what the analyses are showing."""
        return set(self.checked) != set(self.applied)

    def applied_scans(self, scans) -> list:
        """The applied scans, in session order."""
        keep = set(self.applied)
        return [s for s in (scans or []) if getattr(s, "name", None) in keep]

    def to_json(self) -> dict:
        return {"applied": list(self.applied)}

    def load(self, data, available_names) -> None:
        """Restore from a session JSON; names no longer in the session are dropped."""
        avail = set(available_names or [])
        names = [n for n in ((data or {}).get("applied") or []) if n in avail]
        self.checked = list(names)
        self.applied = list(names)


_current = SessionSelection()


def current() -> SessionSelection:
    return _current


def reset() -> None:
    global _current
    _current = SessionSelection()
