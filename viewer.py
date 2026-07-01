from __future__ import annotations

from matplotlib.figure import Figure

try:
    from .pipeline import VIRTUAL_IMAGE_KEYS, load_virtual_images_step
    from .state import WorkflowState
except ImportError:
    from pipeline import VIRTUAL_IMAGE_KEYS, load_virtual_images_step
    from state import WorkflowState


DISPLAY_TITLES = {
    "annular_dark_field": "ADF",
    "bright_field": "BF",
    "dp_mean": "DP mean",
    "dp_max": "DP max",
}


def _apply_r_pixel_to_virtual_images(state: WorkflowState) -> None:
    """
    Propagate real-space pixel calibration to each virtual image object so that
    py4DSTEM.show() displays axes in nm (or the chosen unit) instead of pixels.

    Priority:
      1. state.image_pixel_size / state.image_pixel_units   (set by set_image_pixel_calibration)
      2. datacube.calibration.get_R_pixel_size()             (set on load from the Load Dialog)

    Mirrors the notebook pattern:
        im_adf.calibration.set_R_pixel_size(R_pixel)
        im_adf.calibration.set_R_pixel_units(R_unit)
    """
    if not state.virtual_images:
        return

    # Resolve calibration value from state
    r_px: float | None = getattr(state, "image_pixel_size", None)
    r_units: str = getattr(state, "image_pixel_units", None) or "nm"

    if r_px is None or r_px <= 0:
        # Fall back to datacube.calibration (set by Load Dialog)
        try:
            cal = getattr(getattr(state, "datacube", None), "calibration", None)
            if cal is not None and hasattr(cal, "get_R_pixel_size"):
                v = float(cal.get_R_pixel_size())
                if v > 0:
                    r_px = v
                    r_units = str(cal.get_R_pixel_units() or "nm")
        except Exception:
            pass

    if r_px is None or r_px <= 0:
        return  # no calibration available — leave virtual images as-is

    for key in VIRTUAL_IMAGE_KEYS:
        img_obj = state.virtual_images.get(key)
        if img_obj is None:
            continue
        try:
            cal_obj = getattr(img_obj, "calibration", None)
            if cal_obj is not None and hasattr(cal_obj, "set_R_pixel_size"):
                cal_obj.set_R_pixel_size(r_px)
                cal_obj.set_R_pixel_units(r_units)
        except Exception:
            pass  # never crash the viewer over a calibration propagation failure


def build_virtual_image_figure(state: WorkflowState, title: str | None = None) -> Figure:
    """Build exactly one 2x2 figure for the virtual image viewer."""

    import py4DSTEM

    if not state.virtual_images:
        load_virtual_images_step(state)

    # Propagate R-pixel calibration to each image object so axes show in nm
    _apply_r_pixel_to_virtual_images(state)

    fig = Figure(figsize=(9.5, 8.0), constrained_layout=True)
    axes = fig.subplots(2, 2)

    for ax, key in zip(axes.ravel(), VIRTUAL_IMAGE_KEYS):
        py4DSTEM.show(state.virtual_images[key], figax=(fig, ax), title=DISPLAY_TITLES[key])

    if title:
        fig.suptitle(title)
    return fig
