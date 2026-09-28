"""G2: an ADF with no vacuum must not blank real crystal.

The old "void" rule normalised the ADF between its 15th and 99.5th percentiles and
zeroed every pixel below 6 %: it always cut the darkest part of the image, so a field
with no vacuum lost real crystal (synthetic Si: 3 140 of 16 384 pixels). Vacuum is
already marked by py4DSTEM's fit mask (channel 4), which is what must decide.
"""
import numpy as np

import pipeline as P
import synthetic_si110 as S


class _State:
    def __init__(self, adf):
        self.virtual_images = {"annular_dark_field": adf} if adf is not None else {}


def _fresh_raw(flow):
    """Unsanitized (5, H, W) tensor of the fixture's StrainMap."""
    return P._strain_raw_from_strainmap(flow["scan"].state.strainmap_full)


def _n_valid(raw):
    return int((np.asarray(raw)[4] > 0.5).sum())


def test_low_contrast_adf_without_vacuum_keeps_every_pixel(mini_base):
    adf = S.synthetic_adf(mini_base["truth"], contrast=0.05)
    raw = P._sanitize_strain_raw(_fresh_raw(mini_base), _State(adf), "without_roi")
    assert _n_valid(raw) == mini_base["truth"]["n"] ** 2


def test_vacuum_stays_invalid_with_or_without_adf(mini_vacuum):
    vac = mini_vacuum["truth"]["vacuum"]
    adf = S.synthetic_adf(mini_vacuum["truth"], contrast=0.05)
    for state in (_State(adf), _State(None)):
        raw = np.asarray(P._sanitize_strain_raw(_fresh_raw(mini_vacuum), state, "without_roi"))
        valid = raw[4] > 0.5
        assert np.array_equal(valid, ~vac)
        assert np.all(raw[:, vac] == 0.0)
