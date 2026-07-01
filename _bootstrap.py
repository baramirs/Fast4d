"""Path bootstrap for Fast4d — GUI and pipeline deps both live here (self-contained)."""
from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
ICONS_DIR = APP_ROOT / "icons"


def icon_path(name: str) -> Path:
    """Return ``icons/<name>`` under the app root (adds ``.png`` if missing)."""
    p = Path(name)
    if p.suffix.lower() != ".png":
        p = p.with_suffix(".png")
    return ICONS_DIR / p.name


def bootstrap_sys_path() -> Path:
    """Put Fast4d on ``sys.path``. Returns the pipeline root (now ``APP_ROOT`` itself).

    Fast4d is self-contained: pipeline.py / state.py / fast_artifacts.py /
    fast_batch.py / stress_analysis.py / batch_figures.py / batch_common.py /
    viewer.py / calib_params_io.py all live alongside this file. The legacy
    ``../4DSTEM`` sibling lookup has been removed.
    """
    pipeline_root = APP_ROOT
    s = str(APP_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)
    return pipeline_root
