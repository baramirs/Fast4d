"""choose_basis_vectors may pick any pair of the four equivalent {111} spots
(Scan01 vs Scan02, New batch). With fixed hkl labels, a pair at 70.5 deg
instead of 109.5 deg put x along [001] instead of [-110] (coordinate_rotation
264 vs 175 deg: exx and eyy swapped). resolve_labels fixes the label signs from
the measured angle so x lands on the same crystal line for every scan."""
import crystal_frame as CF

SPEC = dict(CF.DEFAULT_SPEC)
# reference g1, g2 (qx, qy in BVM px) measured on real scans
SCAN02 = ([-12.82, -7.44], [11.12, -9.59])     # 109.1 deg: labels fit
SCAN03 = ([-11.12, 9.59], [12.82, 7.43])       # 109.1 deg: labels fit
SCAN01 = ([11.11, -9.59], [12.82, 7.43])       # 70.9 deg: labels do not fit


def _x_line(g1, g2):
    spec, info = CF.resolve_labels(g1, g2, SPEC)
    fr = CF.frame_from_g(g1, g2, spec)
    return fr["angle_deg"] % 180.0, fr, info


def test_fitting_labels_are_left_unchanged():
    spec, info = CF.resolve_labels(*SCAN02, SPEC)
    assert spec["g1_hkl"] == SPEC["g1_hkl"] and spec["g2_hkl"] == SPEC["g2_hkl"]
    assert info["changed"] is False


def test_scan01_gets_the_same_x_line_as_the_others():
    ref, _, _ = _x_line(*SCAN02)
    x01, fr01, info01 = _x_line(*SCAN01)
    x03, _, _ = _x_line(*SCAN03)
    assert info01["changed"] is True
    assert fr01["label_ok"]
    assert abs(x01 - ref) < 1.0
    assert abs(x03 - ref) < 1.0


def test_resolved_labels_stay_in_zone():
    spec, _ = CF.resolve_labels(*SCAN01, SPEC)
    assert CF.check_spec(spec) == []
