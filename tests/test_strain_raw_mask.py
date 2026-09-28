"""Failed-fit pixels must come out as 0, not as ~100 % strain.

Where py4DSTEM cannot fit the lattice (vacuum, amorphous) g1 = g2 = 0, so
e_xx = e_yy = 1 exactly. After get_strain rotates the tensor those pixels are
cos^2 + sin^2 = 0.9999999..., which slips under the |e| >= 1 artefact rule. The
fit mask (channel 4) is the reliable signal and must zero them.
"""
import numpy as np

import pipeline as P


class _State:
    virtual_images = {}          # no ADF -> no void mask: the fit mask alone must do it


def _stack(bg_value):
    H, W = 12, 16                # > 8 so the channel axis is unambiguous (real maps: 256x256)
    raw = np.zeros((5, H, W))
    raw[0] = raw[1] = 0.002      # eyy, exx inside the crystal
    raw[2] = 0.001               # exy
    raw[3] = 0.01                # theta
    raw[4] = 1.0                 # fit mask: fitted
    raw[:2, :, :3] = bg_value    # failed fit on the left columns
    raw[2:4, :, :3] = 0.0
    raw[4, :, :3] = 0.0
    return raw


def test_failed_fit_pixels_zeroed_even_when_rotation_leaves_them_below_one():
    raw = _stack(1.0 - 1e-15)
    out = np.asarray(P._sanitize_strain_raw(raw, _State(), "without_roi"))
    assert np.all(out[:, :, :3] == 0.0)


def test_fitted_pixels_untouched():
    raw = _stack(1.0 - 1e-15)
    out = np.asarray(P._sanitize_strain_raw(raw, _State(), "without_roi"))
    assert np.allclose(out[0, :, 3:], 0.002)
    assert np.allclose(out[4, :, 3:], 1.0)


def test_channels_last_uses_fit_mask_too():
    raw = np.moveaxis(_stack(1.0 - 1e-15), 0, -1).copy()
    out = np.asarray(P._sanitize_strain_raw(raw, _State(), "without_roi"))
    assert np.all(out[:, :3, :] == 0.0)
