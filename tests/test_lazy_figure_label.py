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
