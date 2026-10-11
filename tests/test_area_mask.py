"""engine.area_mask: user-drawn areas (rotatable rectangle, circle/ellipse, polygon), union and
exclusion, for the "Ellipse from area" tool.

Pixel (row y, column x) is inside when its centre (x, y) lies inside the shape, boundary
included -- the same rule as engine.roi_mask, which the rectangle reuses unchanged.
"""
import numpy as np
import pytest

import engine as E

SHAPE = (40, 50)


def _brute(shape, inside):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.vectorize(inside)(xx.astype(float), yy.astype(float))


def test_rect_equals_roi_mask():
    for b in ([5, 20, 3, 12], [10.2, 30.7, 4.6, 25.1]):
        m = E.area_mask(SHAPE, [{"kind": "rect", "bounds": b}])
        assert np.array_equal(m, E.roi_mask(SHAPE, b))


def test_rotated_rect_equals_roi_mask():
    b = [10, 30, 8, 20]
    m = E.area_mask(SHAPE, [{"kind": "rect", "bounds": b, "angle": 30.0}])
    assert np.array_equal(m, E.roi_mask(SHAPE, b + [30.0]))


def test_axes_x_is_column_y_is_row():
    m = E.area_mask(SHAPE, [{"kind": "rect", "bounds": [0, 2, 5, 6]}])
    expected = np.zeros(SHAPE, bool)
    expected[5:7, 0:3] = True
    assert np.array_equal(m, expected)


def test_circle_counts_lattice_points():
    m = E.area_mask(SHAPE, [{"kind": "ellipse", "center": [20, 15], "radii": [5, 5]}])
    assert m.sum() == 81                                   # integer points with x^2 + y^2 <= 25
    assert m[15, 20] and m[15, 25] and m[10, 20] and not m[15, 26]


def test_rotated_ellipse_matches_brute_force():
    cx, cy, rx, ry, ang = 25.0, 20.0, 12.0, 5.0, 35.0
    t = np.radians(ang)

    def inside(x, y):
        u = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
        v = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
        return (u / rx) ** 2 + (v / ry) ** 2 <= 1 + 1e-9

    m = E.area_mask(SHAPE, [{"kind": "ellipse", "center": [cx, cy], "radii": [rx, ry],
                             "angle": ang}])
    assert np.array_equal(m, _brute(SHAPE, inside))


def test_ellipse_rotated_90_equals_swapped_radii():
    a = E.area_mask(SHAPE, [{"kind": "ellipse", "center": [25, 20], "radii": [10, 4], "angle": 90}])
    b = E.area_mask(SHAPE, [{"kind": "ellipse", "center": [25, 20], "radii": [4, 10]}])
    assert np.array_equal(a, b)


def test_polygon_square_equals_rect_boundary_included():
    sq = [[2, 3], [8, 3], [8, 9], [2, 9]]
    m = E.area_mask(SHAPE, [{"kind": "polygon", "points": sq}])
    assert np.array_equal(m, E.roi_mask(SHAPE, [2, 8, 3, 9]))
    rev = E.area_mask(SHAPE, [{"kind": "polygon", "points": sq[::-1]}])
    assert np.array_equal(m, rev)                          # vertex order does not matter


def test_concave_polygon_matches_brute_force():
    pts = np.array([[5, 5], [30, 5], [30, 30], [18, 12], [5, 30]], float)

    def inside(x, y):
        from matplotlib.path import Path
        on_edge = False
        for (x0, y0), (x1, y1) in zip(pts, np.roll(pts, -1, axis=0)):
            d = np.array([x1 - x0, y1 - y0])
            tt = np.clip(((x - x0) * d[0] + (y - y0) * d[1]) / (d @ d), 0, 1)
            if np.hypot(x - (x0 + tt * d[0]), y - (y0 + tt * d[1])) <= 1e-9:
                on_edge = True
        return on_edge or Path(pts).contains_point((x, y))

    m = E.area_mask(SHAPE, [{"kind": "polygon", "points": pts.tolist()}])
    assert np.array_equal(m, _brute(SHAPE, inside))
    assert not m[25, 18]                                   # the notch is outside


def test_union_and_exclusion():
    a = {"kind": "rect", "bounds": [0, 9, 0, 9]}
    b = {"kind": "rect", "bounds": [20, 29, 20, 29]}
    hole = {"kind": "ellipse", "center": [4, 4], "radii": [2, 2], "exclude": True}
    m = E.area_mask(SHAPE, [a, b])
    assert m.sum() == 200
    m2 = E.area_mask(SHAPE, [a, b, hole])
    assert m2.sum() == 200 - 13 and not m2[4, 4]
    assert np.array_equal(E.area_mask(SHAPE, [hole, a, b]), m2)   # exclusion applies last


def test_outside_and_clipping():
    assert E.area_mask(SHAPE, [{"kind": "ellipse", "center": [500, 500], "radii": [3, 3]}]).sum() == 0
    # a rectangle fully off the image is EMPTY (roi_mask clamps it onto the edge row/column)
    for b in ([500, 520, 500, 520], [-30, -5, 2, 8], [3, 9, 60, 70]):
        assert E.area_mask(SHAPE, [{"kind": "rect", "bounds": b}]).sum() == 0
        assert E.area_mask(SHAPE, [{"kind": "rect", "bounds": b, "angle": 20.0}]).sum() == 0
    part = E.area_mask(SHAPE, [{"kind": "rect", "bounds": [45, 60, 0, 3]}])
    assert part.sum() == 5 * 4                             # columns 45-49, rows 0-3
    m = E.area_mask(SHAPE, [{"kind": "polygon", "points": [[-10, -10], [5, -10], [5, 5], [-10, 5]]}])
    assert m.sum() == 36                                   # columns 0-5, rows 0-5
    assert E.area_mask(SHAPE, []).sum() == 0


def test_single_dict_is_accepted():
    a = {"kind": "rect", "bounds": [1, 3, 1, 3]}
    assert np.array_equal(E.area_mask(SHAPE, a), E.area_mask(SHAPE, [a]))


@pytest.mark.parametrize("bad", [
    {"kind": "polygon", "points": [[0, 0], [5, 5]]},
    {"kind": "ellipse", "center": [5, 5], "radii": [0, 3]},
    {"kind": "rect", "bounds": [1, 2, 3]},
    {"kind": "star"},
])
def test_invalid_areas_raise(bad):
    with pytest.raises(ValueError):
        E.area_mask(SHAPE, [bad])
