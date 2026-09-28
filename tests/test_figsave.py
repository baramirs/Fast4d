"""figsave: a worker thread must never call savefig on a figure the GUI shows.

savefig(dpi=300) temporarily sets fig.dpi = 300; a Qt resize at that moment
sizes the figure for 300 dpi and it stays shrunk to ~1/3 (the "third file"
strain window). Worker-thread saves are queued and written by flush() on the
main thread.
"""
import threading

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

import figsave


def _fig():
    fig = Figure(figsize=(4, 3), dpi=100)
    fig.add_subplot(111).plot([0, 1], [0, 1])
    return fig


def test_main_thread_saves_immediately(tmp_path):
    p = tmp_path / "a.png"
    assert figsave.savefig(_fig(), p, dpi=150) is True
    assert p.is_file()


def test_worker_thread_defers_and_does_not_touch_the_figure(tmp_path):
    fig = _fig()
    p = tmp_path / "b.png"
    seen = {}

    def worker():
        dpi_before = fig.dpi
        seen["ret"] = figsave.savefig(fig, p, dpi=300, bbox_inches="tight")
        seen["dpi_unchanged"] = fig.dpi == dpi_before

    t = threading.Thread(target=worker)
    t.start(); t.join()
    assert seen == {"ret": False, "dpi_unchanged": True}
    assert not p.exists()
    assert figsave.pending_count() == 1
    assert figsave.flush() == 1
    assert p.is_file()
    assert figsave.pending_count() == 0


def test_flush_reports_failures_and_keeps_going(tmp_path):
    good = tmp_path / "ok.png"
    bad = tmp_path / "missing_dir" / "x" / "bad.png"

    def worker():
        figsave.savefig(_fig(), bad, dpi=72)
        figsave.savefig(_fig(), good, dpi=72)

    t = threading.Thread(target=worker)
    t.start(); t.join()
    msgs = []
    assert figsave.flush(log=msgs.append) == 1
    assert good.is_file()
    assert any("bad.png" in m for m in msgs)
