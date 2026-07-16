"""Indexer plugin protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from plugins.indexing.types import BasisProposal, BvmContext


@runtime_checkable
class IndexerPlugin(Protocol):
    """Runnable BVM → BasisProposal indexer (no strain pipeline side effects)."""

    id: str
    label: str
    description: str

    def run(self, ctx: BvmContext, *, log=None) -> BasisProposal:
        ...
