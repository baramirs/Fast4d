"""Rotated ROI mask, line direction and relative drift (engine geometry)."""
import numpy as np

import engine as E


def _old_rect_mask(shape, b):
    H, W = shape
    x0, x1, y0, y1 = (int(round(float(v))) for v in b)
    x0, x1 = sorted((max(0, min(x0, W - 1)), max(0, min(x1, W - 1))))
    y0, y1 = sorted((max(0, min(y0, H - 1)), max(0, min(y1, H - 1))))
    m = np.zeros(shape, bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def test_angle_zero_equals_current_rectangle():
    for b in ([3, 10, 5, 9], [0.4, 7.6, 2.2, 2.9], [-5, 4, 30, 50]):
        assert np.array_equal(E.roi_mask((40, 30), b), _old_rect_mask((40, 30), b))
        assert np.array_equal(E.roi_mask((40, 30), list(b) + [0.0]),
                              _old_rect_mask((40, 30), b))


def test_ninety_degrees_swaps_sides():
    m0 = E.roi_mask((60, 60), [20, 40, 26, 34])          # 21 wide x 9 tall
    m90 = E.roi_mask((60, 60), [20, 40, 26, 34, 90.0])
    ys, xs = np.nonzero(m90)
    assert xs.max() - xs.min() + 1 == 9
    assert ys.max() - ys.min() + 1 == 21
    assert m0.sum() == m90.sum()


def test_rotated_area_close_to_length_times_width():
    m = E.roi_mask((200, 200), [60, 140, 80, 120, 30.0])  # 80 x 40
    assert abs(m.sum() - 81 * 41) / (81 * 41) < 0.05


def test_region_values_use_the_mask():
    a = np.arange(60 * 60, dtype=float).reshape(60, 60)
    roi = [20, 40, 26, 34, 90.0]
    v = E._roi_region_values(a, roi)
    assert np.allclose(np.sort(v), np.sort(a[E.roi_mask(a.shape, roi)]))


def test_normalize_keeps_angle():
    assert E._normalize_roi([10, 2, 3, 8]) == [2.0, 10.0, 3.0, 8.0]
    assert E._normalize_roi([10, 2, 3, 8, 0.0]) == [2.0, 10.0, 3.0, 8.0]
    assert E._normalize_roi([10, 2, 3, 8, 25.0]) == [2.0, 10.0, 3.0, 8.0, 25.0]


def test_shift_clamp_keeps_angle():
    assert E._roi_shift_clamp([2, 10, 3, 8, 25.0], 50, 50, 1, 1) == [3, 11, 4, 9, 25.0]
    assert E._roi_shift_clamp([2, 10, 3, 8], 50, 50, 1, 1) == [3, 11, 4, 9]


def test_reversed_line_reverses_profile():
    a = np.tile(np.arange(50, dtype=float), (20, 1))      # ramp along x
    d, v = E._line_samples(a, [[5, 10], [40, 10]], width=3)
    d2, v2 = E._line_samples(a, [[40, 10], [5, 10]], width=3)
    assert np.allclose(v, v2[::-1])
    assert v[0] < v[-1]                                   # distance starts at the first point


def test_relative_drift_uses_template_drift():
    tpl, s = E.Scan(name="t"), E.Scan(name="s")
    tpl.drift, s.drift = (2.0, -1.0), (5.0, 3.0)
    assert E.relative_drift_shift(s, tpl, use_drift=True) == (-3.0, -4.0)
    assert E.relative_drift_shift(tpl, tpl, use_drift=True) == (0.0, 0.0)
    assert E.relative_drift_shift(s, tpl, use_drift=False) == (0.0, 0.0)
    tpl.drift = None                                      # template without drift = reference
    assert E.relative_drift_shift(s, tpl, use_drift=True) == (-5.0, -3.0)


def test_place_roi_relative_to_template(monkeypatch):
    tpl, s = E.Scan(name="t"), E.Scan(name="s")
    tpl.drift, s.drift = (2.0, 0.0), (5.0, 0.0)
    monkeypatch.setattr(E, "_scan_shape", lambda sc: (50, 50))
    E.place_roi_from_spec([tpl, s], "R1", [10, 20, 10, 20, 15.0], template_scan=tpl,
                          use_drift=True)
    assert E.scan_area_rois(tpl)["R1"] == [10.0, 20.0, 10.0, 20.0, 15.0]
    assert E.scan_area_rois(s)["R1"] == [7.0, 17.0, 10.0, 20.0, 15.0]


def test_line_width_clamped_to_200():
    assert E.clamp_line_width(500) == 200
    assert E.clamp_line_width(0) == 1
    assert E.clamp_line_width(7.4) == 7
