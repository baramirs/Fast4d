"""measure_model -- Qt-free state and numbers of the Line / ROI profiles tool.

Elements are drawn on a *template* file and sampled on every checked file:

* line  ``geom = [[x0, y0], [x1, y1]]`` -- the FIRST point is the start; the
  profile distance is measured from it. ``width`` = integration width in px
  (1..engine.LINE_WIDTH_MAX).
* roi   ``geom = [x0, x1, y0, y1, angle_deg]`` -- rectangle rotated about its
  centre (engine.roi_mask is the pixel definition).

An element moves onto another file by ``engine.relative_drift_shift`` (drift of
that file minus the template's). Units are nm when the template file has a scan
step (``params.r_px_pm`` > 0), px otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import engine as E

COLORS = ["#00E5FF", "#FF6D00", "#76FF03", "#E040FB", "#FFEA00", "#18FFFF",
          "#FF4081", "#B2FF59"]


@dataclass
class Element:
    id: str
    kind: str                     # "line" | "roi"
    geom: list
    width: int = 3                # lines only (px)
    color: str = "#00E5FF"


@dataclass
class MeasureModel:
    elements: list = field(default_factory=list)
    selected: str | None = None
    _count: dict = field(default_factory=lambda: {"line": 0, "roi": 0})

    # ── elements ─────────────────────────────────────────────────────────────
    def _next_id(self, kind: str) -> str:
        taken = {e.id for e in self.elements}
        prefix = "L" if kind == "line" else "R"
        while True:
            self._count[kind] += 1
            eid = f"{prefix}{self._count[kind]}"
            if eid not in taken:
                return eid

    def _color(self) -> str:
        return COLORS[len(self.elements) % len(COLORS)]

    def add_line(self, shape, geom=None, width: int = 3, eid: str | None = None) -> Element:
        H, W = shape
        if geom is None:
            k = sum(1 for e in self.elements if e.kind == "line")
            y = H * (0.35 + 0.12 * (k % 4))
            geom = [[W * 0.15, y], [W * 0.85, y]]
        el = Element(eid or self._next_id("line"), "line",
                     [[float(geom[0][0]), float(geom[0][1])],
                      [float(geom[1][0]), float(geom[1][1])]],
                     E.clamp_line_width(width), self._color())
        self.elements.append(el)
        self.selected = el.id
        return el

    def add_roi(self, shape, geom=None, eid: str | None = None) -> Element:
        H, W = shape
        if geom is None:
            k = sum(1 for e in self.elements if e.kind == "roi")
            x0, y0 = W * (0.1 + 0.12 * (k % 4)), H * (0.1 + 0.12 * (k % 4))
            geom = [x0, x0 + W * 0.25, y0, y0 + H * 0.25, 0.0]
        g = [float(v) for v in geom[:4]] + [E.roi_angle(geom)]
        el = Element(eid or self._next_id("roi"), "roi", g, 3, self._color())
        self.elements.append(el)
        self.selected = el.id
        return el

    def get(self, eid) -> Element | None:
        return next((e for e in self.elements if e.id == eid), None)

    def selected_element(self) -> Element | None:
        return self.get(self.selected)

    def select(self, eid) -> None:
        if self.get(eid) is not None:
            self.selected = eid

    def delete_selected(self) -> str | None:
        """Remove ONLY the selected element; the others keep id and geometry."""
        el = self.selected_element()
        if el is None:
            return None
        i = self.elements.index(el)
        self.elements.remove(el)
        self.selected = (self.elements[min(i, len(self.elements) - 1)].id
                         if self.elements else None)
        return el.id

    def clear(self) -> None:
        self.elements.clear()
        self.selected = None

    def lines(self) -> list:
        return [e for e in self.elements if e.kind == "line"]

    def rois(self) -> list:
        return [e for e in self.elements if e.kind == "roi"]

    def set_width(self, eid, width) -> int:
        el = self.get(eid)
        w = E.clamp_line_width(width)
        if el is not None:
            el.width = w
        return w


# ── units and properties ──────────────────────────────────────────────────────
def nm_per_px(scan) -> float | None:
    """Scan step in nm/px from ``params.r_px_pm`` (None = no step → px)."""
    pm = float(getattr(getattr(scan, "params", None), "r_px_pm", 0.0) or 0.0)
    return pm / 1000.0 if pm > 0 else None


def unit_of(scan) -> str:
    return "nm" if nm_per_px(scan) else "px"


def line_props(geom) -> dict:
    (x0, y0), (x1, y1) = geom
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "length": float(math.hypot(x1 - x0, y1 - y0))}


def line_with_length(geom, length: float) -> list:
    """Keep the start and the direction, set the length (px)."""
    (x0, y0), (x1, y1) = geom
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = ((x1 - x0) / L, (y1 - y0) / L) if L > 1e-9 else (1.0, 0.0)
    return [[x0, y0], [x0 + ux * float(length), y0 + uy * float(length)]]


def roi_props(geom) -> dict:
    x0, x1, y0, y1 = geom[:4]
    return {"cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0,
            "length": x1 - x0, "width": y1 - y0, "angle": E.roi_angle(geom)}


def roi_from_props(cx, cy, length, width, angle) -> list:
    hl, hw = abs(float(length)) / 2.0, abs(float(width)) / 2.0
    return [cx - hl, cx + hl, cy - hw, cy + hw, float(angle)]


# ── drift ─────────────────────────────────────────────────────────────────────
def any_drift(scans) -> bool:
    return any(getattr(s, "drift", None) for s in (scans or []))


def drift_signature(scans) -> tuple:
    return tuple((getattr(s, "name", ""), tuple(getattr(s, "drift", None) or ()))
                 for s in (scans or []))


def shifted_line(geom, dx, dy) -> list:
    return [[geom[0][0] + dx, geom[0][1] + dy], [geom[1][0] + dx, geom[1][1] + dy]]


def shifted_roi(geom, dx, dy) -> list:
    return [geom[0] + dx, geom[1] + dx, geom[2] + dy, geom[3] + dy, E.roi_angle(geom)]


# ── numbers ───────────────────────────────────────────────────────────────────
def line_curves(scans, template, lines, channel, label, *, use_drift, get_map=None) -> list:
    """[{line, scan, dist_px, values}] for every (line, file). The segment is NOT
    clamped: bilinear sampling uses the nearest edge value outside the map."""
    get_map = get_map or E.channel_map_2d
    out = []
    for el in lines:
        for sc in scans:
            m = get_map(sc, channel, label)
            if m is None:
                continue
            dx, dy = E.relative_drift_shift(sc, template, use_drift=use_drift)
            dist, vals = E._line_samples(np.asarray(m, dtype=float),
                                         shifted_line(el.geom, dx, dy), width=el.width)
            out.append({"line": el.id, "scan": sc.name, "dist_px": np.asarray(dist),
                        "values": np.asarray(vals)})
    return out


def roi_rows(scans, template, rois, channel, label, *, use_drift, get_map=None) -> list:
    """[{roi, scan, mean, sd, n, values}] -- the pixels of each ROI on each file."""
    get_map = get_map or E.channel_map_2d
    out = []
    for el in rois:
        for sc in scans:
            m = get_map(sc, channel, label)
            if m is None:
                continue
            dx, dy = E.relative_drift_shift(sc, template, use_drift=use_drift)
            v = E._roi_region_values(np.asarray(m, dtype=float), shifted_roi(el.geom, dx, dy))
            n = int(v.size)
            out.append({"roi": el.id, "scan": sc.name, "values": v, "n": n,
                        "mean": float(np.mean(v)) if n else float("nan"),
                        "sd": float(np.std(v, ddof=1)) if n > 1 else float("nan")})
    return out


def histogram(rows, bins: int = 40):
    """Common bins over the given ROI rows → (centers, {scan: counts})."""
    allv = np.concatenate([r["values"] for r in rows if r["n"]]) if rows else np.zeros(0)
    if allv.size == 0:
        return np.zeros(0), {}
    edges = np.histogram_bin_edges(allv, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, {r["scan"]: np.histogram(r["values"], bins=edges)[0] for r in rows}


# ── clipboard (tab separated, header with units) ─────────────────────────────
def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "" if not np.isfinite(f) else f"{f:.6g}"


def profile_text(curves, dist_unit: str, value_unit: str, scale: float = 1.0) -> str:
    """distance column + one column per curve ``<line> <file> (<unit>)``. Curves of
    different length share the distance column of the longest one."""
    if not curves:
        return ""
    longest = max(curves, key=lambda c: len(c["dist_px"]))
    head = [f"distance ({dist_unit})"] + [f"{c['line']} {c['scan']} ({value_unit})"
                                          for c in curves]
    rows = ["\t".join(head)]
    for i, d in enumerate(longest["dist_px"]):
        cells = [_fmt(d * scale)]
        for c in curves:
            cells.append(_fmt(c["values"][i]) if i < len(c["values"]) else "")
        rows.append("\t".join(cells))
    return "\n".join(rows)


def roi_text(rows, value_unit: str) -> str:
    out = [f"ROI\tfile\tmean ({value_unit})\tsd ({value_unit})\tn"]
    for r in rows:
        out.append("\t".join([r["roi"], r["scan"], _fmt(r["mean"]), _fmt(r["sd"]),
                              str(r["n"])]))
    return "\n".join(out)


def histogram_text(centers, counts: dict, value_unit: str) -> str:
    names = list(counts)
    out = ["\t".join([f"bin_center ({value_unit})"] + [f"{n} (counts)" for n in names])]
    for i, c in enumerate(centers):
        out.append("\t".join([_fmt(c)] + [str(int(counts[n][i])) for n in names]))
    return "\n".join(out)
