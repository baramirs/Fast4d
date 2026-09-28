"""Line / ROI tool model: ids, delete isolation, width clamp, drift, Ctrl+C text."""
import numpy as np

import engine as E
import measure_model as M


def _scan(name, arr, drift=None):
    s = E.Scan(name=name)
    s.drift = drift
    s._arr = arr
    return s


def _get_map(sc, ch, lab):
    return sc._arr


def test_ids_and_delete_only_selected():
    m = M.MeasureModel()
    l1 = m.add_line((50, 60))
    l2 = m.add_line((50, 60))
    r1 = m.add_roi((50, 60))
    g2 = [list(p) for p in l2.geom]
    assert [e.id for e in m.elements] == ["L1", "L2", "R1"]
    m.select("L1")
    assert m.delete_selected() == "L1"
    assert [e.id for e in m.elements] == ["L2", "R1"]
    assert m.get("L2").geom == g2 and m.get("R1").geom == r1.geom
    assert m.add_line((50, 60)).id == "L3"             # ids are not reused
    assert l1.id not in {e.id for e in m.elements}


def test_width_clamped():
    m = M.MeasureModel()
    el = m.add_line((50, 60), width=999)
    assert el.width == 200
    assert m.set_width(el.id, 0) == 1


def test_profile_direction_follows_start():
    a = np.tile(np.arange(60, dtype=float), (40, 1))
    s = _scan("a", a)
    m = M.MeasureModel()
    fwd = m.add_line((40, 60), geom=[[5, 10], [50, 10]])
    rev = m.add_line((40, 60), geom=[[50, 10], [5, 10]])
    c = M.line_curves([s], s, [fwd, rev], "eyy", "without_roi", use_drift=False,
                      get_map=_get_map)
    assert np.allclose(c[0]["values"], c[1]["values"][::-1])
    assert c[0]["values"][0] < c[0]["values"][-1]


def test_relative_drift_moves_the_line():
    a = np.tile(np.arange(60, dtype=float), (40, 1))   # value = x
    tpl = _scan("t", a, drift=(2.0, 0.0))
    s = _scan("s", a, drift=(7.0, 0.0))
    m = M.MeasureModel()
    el = m.add_line((40, 60), geom=[[20, 10], [30, 10]], width=1)
    c = M.line_curves([tpl, s], tpl, [el], "eyy", "without_roi", use_drift=True,
                      get_map=_get_map)
    assert np.isclose(c[0]["values"][0], 20.0)
    assert np.isclose(c[1]["values"][0], 15.0)         # -(7 - 2)


def test_roi_rows_and_clipboard_texts():
    a = np.arange(40 * 60, dtype=float).reshape(40, 60)
    s1, s2 = _scan("f1", a), _scan("f2", a + 1.0)
    m = M.MeasureModel()
    r = m.add_roi((40, 60), geom=[10, 20, 5, 9, 0.0])
    rows = M.roi_rows([s1, s2], s1, [r], "eyy", "without_roi", use_drift=False,
                      get_map=_get_map)
    assert rows[0]["n"] == 11 * 5
    txt = M.roi_text(rows, "%")
    assert txt.splitlines()[0] == "ROI\tfile\tmean (%)\tsd (%)\tn"
    assert txt.splitlines()[2].startswith("R1\tf2\t")
    centers, counts = M.histogram(rows, bins=5)
    h = M.histogram_text(centers, counts, "%").splitlines()
    assert h[0] == "bin_center (%)\tf1 (counts)\tf2 (counts)"
    assert len(h) == 6


def test_profile_clipboard_header_units_and_values():
    a = np.tile(np.arange(60, dtype=float), (40, 1))
    s = _scan("scan02", a)
    m = M.MeasureModel()
    el = m.add_line((40, 60), geom=[[0, 5], [10, 5]], width=1)
    c = M.line_curves([s], s, [el], "eyy", "without_roi", use_drift=False, get_map=_get_map)
    lines = M.profile_text(c, "nm", "%", scale=0.5).splitlines()
    assert lines[0] == "distance (nm)\tL1 scan02 (%)"
    assert lines[1] == "0\t0"
    assert lines[-1] == "5\t10"


def test_props_roundtrip():
    g = M.roi_from_props(30, 20, 16, 8, 25.0)
    p = M.roi_props(g)
    assert (p["cx"], p["cy"], p["length"], p["width"], p["angle"]) == (30, 20, 16, 8, 25.0)
    seg = M.line_with_length([[0, 0], [3, 4]], 10)
    assert np.allclose(seg[1], [6, 8])


def test_units_from_scan_step():
    s = E.Scan(name="a")
    assert M.unit_of(s) == "px"
    s.params.r_px_pm = 143.5
    assert M.unit_of(s) == "nm" and np.isclose(M.nm_per_px(s), 0.1435)
