"""M5: strain figures of a uniaxial field must not come out blank.

_apply_get_strain_vrange_to_figures guessed whether each panel holds fractions or
percent by comparing the panel's max|value| with max|exx| only. With exx ~ 0 and
eyy ~ 0.09 (uniaxial strain, synthetic Si) the eyy panel looked like "percent" and got
vrange +-5 applied to fraction data (+-500 %): blank panels. Each panel is now compared
with its own channel of the tensor.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pipeline as P  # noqa: E402


class _State:
    strain_params = {"get_strain": {"vrange": [-5.0, 5.0], "vrange_theta": [-1.0, 1.0]}}


def _figure(raw, scale):
    fig, axs = plt.subplots(1, 3)
    for ax, (title, ch) in zip(axs, (("e_xx", 1), ("e_yy", 0), ("e_xy", 2))):
        ax.imshow(np.asarray(raw[ch], float) * scale)
        ax.set_title(title)
    return fig


def _clims(fig):
    return {ax.get_title(): tuple(ax.images[0].get_clim()) for ax in fig.axes if ax.images}


def test_uniaxial_fraction_maps_get_fraction_limits(mini_base):
    raw = np.asarray(mini_base["scan"].state.strain_raw["without_roi"], float)
    assert np.nanmax(np.abs(raw[0])) / np.nanmax(np.abs(raw[1])) > 25     # really uniaxial
    fig = _figure(raw, 1.0)
    try:
        P._apply_get_strain_vrange_to_figures(_State(), [fig.number], raw_tensor=raw)
        for title, lim in _clims(fig).items():
            assert np.allclose(lim, (-0.05, 0.05)), (title, lim)
    finally:
        plt.close(fig)


def test_percent_maps_keep_percent_limits(mini_base):
    raw = np.asarray(mini_base["scan"].state.strain_raw["without_roi"], float)
    fig = _figure(raw, 100.0)
    try:
        P._apply_get_strain_vrange_to_figures(_State(), [fig.number], raw_tensor=raw)
        for title, lim in _clims(fig).items():
            assert np.allclose(lim, (-5.0, 5.0)), (title, lim)
    finally:
        plt.close(fig)
