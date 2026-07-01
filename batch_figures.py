"""Figure notebook for batch workflow (one tab per step × scan, renamable)."""
from __future__ import annotations

from matplotlib.figure import Figure

try:
    from .batch_common import batch_figure_title
except ImportError:
    from batch_common import batch_figure_title

# tkinter / tkagg are only needed by BatchFigureNotebook below; importing this module
# just for the plain-matplotlib helpers/constants above must not require Tk to be
# installed (R-1: the q-pixel step in pipeline.py imports only those, not the notebook).
tk = ttk = simpledialog = None
FigureCanvasTkAgg = NavigationToolbar2Tk = None


def _ensure_tk_imports() -> None:
    global tk, ttk, simpledialog, FigureCanvasTkAgg, NavigationToolbar2Tk
    if tk is not None:
        return
    import tkinter as tk_mod
    from tkinter import simpledialog as simpledialog_mod
    from tkinter import ttk as ttk_mod

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _Canvas
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as _Toolbar

    tk = tk_mod
    ttk = ttk_mod
    simpledialog = simpledialog_mod
    FigureCanvasTkAgg = _Canvas
    NavigationToolbar2Tk = _Toolbar

# Fixed sizes for batch embedding (avoid stretched ADF / clipped Q-axis labels).
BATCH_FIG_DPI = 96
BATCH_Q_OVERLAY_SIZE = (8.6, 6.2)
BATCH_ROI_FIG_SIZE = (6.2, 6.2)


def figure_has_images(fig: Figure) -> bool:
    try:
        return any(ax.images for ax in fig.axes)
    except Exception:
        return False


def prepare_batch_display_figure(fig: Figure, *, lock_image_aspect: bool = False) -> None:
    """Keep ADF/ROI aspect ratio; tighten margins so axis labels stay visible."""
    try:
        fig.set_dpi(BATCH_FIG_DPI)
    except Exception:
        pass
    for ax in list(fig.axes):
        try:
            if lock_image_aspect and ax.images:
                ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=9)
        except Exception:
            pass
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.tight_layout(pad=1.15)
        except Exception:
            pass


def apply_batch_figure_suptitle(figure: Figure, step_id: str, tab_title: str) -> None:
    """Place suptitle without overlapping subplot titles (esp. origin 2×3 grids)."""
    sid = (step_id or "").strip()
    naxes = len(figure.axes)
    try:
        if sid.startswith("step9") and naxes >= 4:
            try:
                figure.set_constrained_layout_pads(
                    w_pad=0.02,
                    h_pad=0.03,
                    h_pad_top=0.11,
                    w_pad_inches=0.12,
                    h_pad_inches=0.42,
                )
            except Exception:
                try:
                    figure.subplots_adjust(top=0.90, hspace=0.38, wspace=0.28)
                except Exception:
                    pass
            figure.suptitle(tab_title, fontsize=10, y=1.0)
            return
        if sid.startswith("step12") and naxes >= 2:
            try:
                figure.set_constrained_layout_pads(h_pad_top=0.10, h_pad_inches=0.35)
            except Exception:
                pass
            figure.suptitle(tab_title, fontsize=10, y=1.0)
            return
        figure.suptitle(tab_title, fontsize=10, y=0.98)
    except Exception:
        pass


def finalize_q_pixel_scattering_figure(fig: Figure) -> None:
    """Q pixel overlay / refit plots: room for k and |g| axis labels in batch tabs."""
    try:
        fig.set_size_inches(*BATCH_Q_OVERLAY_SIZE, forward=True)
        fig.set_dpi(BATCH_FIG_DPI)
    except Exception:
        pass
    for ax in list(fig.axes):
        try:
            ax.tick_params(labelsize=9, pad=3)
            xl = ax.get_xlabel()
            yl = ax.get_ylabel()
            if xl:
                ax.set_xlabel(xl, fontsize=9)
            if yl:
                ax.set_ylabel(yl, fontsize=9)
        except Exception:
            pass
    # Prefer the constrained layout engine; only fall back to manual spacing if it
    # isn't available. Calling subplots_adjust/tight_layout AFTER a layout engine is
    # set is incompatible and makes matplotlib warn + ignore them.
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.90)
        except Exception:
            pass
        try:
            fig.tight_layout(pad=1.2, rect=(0, 0, 1, 0.96))
        except Exception:
            pass


class BatchFigureNotebook:
    """Manage matplotlib figures inside a ``ttk.Notebook`` (batch panel)."""

    def __init__(self, notebook: ttk.Notebook) -> None:
        _ensure_tk_imports()
        self._nb = notebook
        self._entries: dict[str, dict] = {}

    def _sanitize_key(self, key: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:120]

    def _prepare_for_display(self, figure: Figure) -> None:
        prepare_batch_display_figure(figure, lock_image_aspect=figure_has_images(figure))
        if not figure_has_images(figure):
            finalize_q_pixel_scattering_figure(figure)

    def show_figure(
        self,
        step_id: str,
        scan_stem: str,
        figure: Figure,
        *,
        title: str | None = None,
        select: bool = True,
    ) -> str:
        tab_title = (title or batch_figure_title(step_id, scan_stem)).strip()
        key = self._sanitize_key(f"{step_id}::{scan_stem}::{tab_title}")
        self._prepare_for_display(figure)
        apply_batch_figure_suptitle(figure, step_id, tab_title)
        entry = self._entries.get(key)
        if entry is None:
            frame = ttk.Frame(self._nb)
            toolbar_row = ttk.Frame(frame)
            toolbar_row.pack(fill=tk.X)
            plot_host = ttk.Frame(frame)
            plot_host.pack(fill=tk.BOTH, expand=True)
            canvas = FigureCanvasTkAgg(figure, master=plot_host)
            try:
                NavigationToolbar2Tk(canvas, toolbar_row, pack_toolbar=True).update()
            except Exception:
                pass
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._nb.add(frame, text=tab_title[:48])
            entry = {
                "frame": frame,
                "canvas": canvas,
                "figure": figure,
                "title": tab_title,
                "step_id": step_id,
                "scan_stem": scan_stem,
            }
            self._entries[key] = entry
        else:
            entry["figure"] = figure
            entry["canvas"].figure = figure
            self._prepare_for_display(figure)
            entry["canvas"].draw()
            entry["title"] = tab_title
            try:
                idx = self._nb.index(entry["frame"])
                self._nb.tab(idx, text=tab_title[:48])
            except Exception:
                pass
        if select:
            try:
                self._nb.select(entry["frame"])
            except Exception:
                pass
        return key

    def show_figure_group(
        self,
        step_id: str,
        scan_stem: str,
        figures: list,
        *,
        title: str | None = None,
        panel_titles: list[str] | None = None,
        select: bool = True,
    ) -> str:
        """Stack several matplotlib figures in one scrollable batch tab (e.g. Step 12 basis panels)."""
        figs = [f for f in figures if f is not None]
        if not figs:
            return ""
        tab_title = (title or batch_figure_title(step_id, scan_stem)).strip()
        key = self._sanitize_key(f"{step_id}::{scan_stem}::group::{tab_title}")

        entry = self._entries.get(key)
        if entry is not None:
            try:
                self._nb.forget(entry["frame"])
            except Exception:
                pass
            self._entries.pop(key, None)

        frame = ttk.Frame(self._nb)
        toolbar_row = ttk.Frame(frame)
        toolbar_row.pack(fill=tk.X)
        outer = ttk.Frame(frame)
        outer.pack(fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        cvs = tk.Canvas(outer, highlightthickness=0, yscrollcommand=sb.set)
        sb.configure(command=cvs.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(cvs)
        win = cvs.create_window((0, 0), window=inner, anchor="nw")

        def _cfg_inner(_e=None) -> None:
            try:
                cvs.configure(scrollregion=cvs.bbox("all"))
            except Exception:
                pass

        def _cfg_canvas(e) -> None:
            try:
                cvs.itemconfigure(win, width=int(e.width))
            except Exception:
                pass

        inner.bind("<Configure>", _cfg_inner)
        cvs.bind("<Configure>", _cfg_canvas)

        def _wheel(event, c=cvs) -> None:
            try:
                if event.num == 4:
                    c.yview_scroll(-1, "units")
                elif event.num == 5:
                    c.yview_scroll(1, "units")
                else:
                    c.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            cvs.bind(seq, _wheel)
            inner.bind(seq, _wheel)

        canvases: list[FigureCanvasTkAgg] = []
        for idx, fig in enumerate(figs, start=1):
            self._prepare_for_display(fig)
            if len(figs) > 1:
                cap = f"Panel {idx}"
                if panel_titles is not None and idx - 1 < len(panel_titles):
                    cap = str(panel_titles[idx - 1])
                ttk.Label(inner, text=cap, font=("Segoe UI", 9, "bold")).pack(
                    anchor="w", padx=4, pady=(6, 2)
                )
            plot_host = ttk.Frame(inner)
            plot_host.pack(fill=tk.X, padx=4, pady=(0, 8))
            canvas = FigureCanvasTkAgg(fig, master=plot_host)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.X)
            canvases.append(canvas)

        try:
            if canvases:
                NavigationToolbar2Tk(canvases[0], toolbar_row, pack_toolbar=True).update()
        except Exception:
            pass

        self._nb.add(frame, text=tab_title[:48])
        entry = {
            "frame": frame,
            "canvas": canvases[0] if canvases else None,
            "figure": figs[0],
            "figures": figs,
            "title": tab_title,
            "step_id": step_id,
            "scan_stem": scan_stem,
        }
        self._entries[key] = entry
        if select:
            try:
                self._nb.select(frame)
            except Exception:
                pass
        return key

    def iter_entries_for_scan(self, scan_stem: str):
        """Yield figure entries whose ``scan_stem`` matches (for batch export)."""
        stem = str(scan_stem).strip()
        for entry in self._entries.values():
            if str(entry.get("scan_stem", "")) == stem:
                yield entry

    def bind_rename_on_double_click(self, parent: tk.Misc) -> None:
        self._nb.bind("<Double-Button-1>", self._on_tab_double_click)

    def _on_tab_double_click(self, event) -> None:
        try:
            idx = self._nb.index(f"@{event.x},{event.y}")
        except Exception:
            return
        try:
            current = str(self._nb.tab(idx, "text"))
        except Exception:
            return
        new = simpledialog.askstring(
            "Rename figure tab",
            "Tab title:",
            initialvalue=current,
            parent=self._nb.winfo_toplevel(),
        )
        if not new or not str(new).strip():
            return
        new_title = str(new).strip()
        self._nb.tab(idx, text=new_title[:48])
        for entry in self._entries.values():
            try:
                if self._nb.index(entry["frame"]) == idx:
                    entry["title"] = new_title
                    break
            except Exception:
                continue
