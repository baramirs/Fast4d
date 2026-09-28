"""M4: a failing mask_in_R must not silently fall back to the whole scan.

pipeline._braggpeaks_for_roi_from_bp caught every exception from
``braggpeaks.mask_in_R`` and returned ALL the data, so a ROI-based Q-pixel or
ellipse fit ran over the whole scan without telling anybody.
"""
import numpy as np
import pytest

import pipeline as P
import synthetic_si110 as S


class _State:
    def __init__(self, roi_mask):
        self.roi_mask = roi_mask


@pytest.fixture(scope="module")
def bv():
    return S.make_braggvectors(12)[0]


def test_bad_roi_mask_raises_instead_of_using_all_data(bv):
    with pytest.raises(RuntimeError, match="ROI"):
        P._braggpeaks_for_roi_from_bp(bv, _State(np.ones((5, 7), bool)), use_roi=True)


def test_good_roi_mask_keeps_only_the_roi(bv):
    m = np.zeros((12, 12), bool)
    m[2:6, 3:9] = True
    out = P._braggpeaks_for_roi_from_bp(bv, _State(m), use_roi=True)
    assert out is not bv
    n_in = sum(out.raw[i, j].data.shape[0] > 0 for i in range(12) for j in range(12))
    assert n_in == int(m.sum())


def test_no_roi_or_use_roi_false_returns_all_data(bv):
    assert P._braggpeaks_for_roi_from_bp(bv, _State(None), use_roi=True) is bv
    assert P._braggpeaks_for_roi_from_bp(bv, _State(np.ones((12, 12), bool)), use_roi=False) is bv
