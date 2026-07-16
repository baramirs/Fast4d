"""Registry of indexing plugins for the Plugin menu."""
from __future__ import annotations

from plugins.indexing.index_bvm_known import IndexBvmKnownPlugin
from plugins.indexing.index_bvm_unknown import IndexBvmUnknownPlugin
from plugins.indexing.orient_peaks import OrientPeaksPlugin
from plugins.indexing.protocol import IndexerPlugin

_PLUGINS: list[IndexerPlugin] = [
    IndexBvmUnknownPlugin(),
    IndexBvmKnownPlugin(),
    OrientPeaksPlugin(),
]


def list_plugins() -> list[IndexerPlugin]:
    return list(_PLUGINS)


def get_plugin(plugin_id: str) -> IndexerPlugin:
    for p in _PLUGINS:
        if p.id == plugin_id:
            return p
    raise KeyError(f"Unknown indexing plugin: {plugin_id!r}")
