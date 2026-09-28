"""ROI-mean diffraction pattern at native detector resolution (no Q resample).

For CrystBox SAED/NBED: export a TIFF whose ``(Qy, Qx)`` matches the camera,
and report Q as nm⁻¹/px (Fast4D stores Å⁻¹/px → multiply by 10).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_A_TO_NM_INV = 10.0  # 1 Å⁻¹ = 10 nm⁻¹


def q_Ainv_to_nminv(q_Ainv_per_px: float) -> float:
    """Convert Fast4D Q (Å⁻¹/px) to CrystBox units (nm⁻¹/px)."""
    return float(q_Ainv_per_px) * _A_TO_NM_INV


def roi_to_slices(
    bounds: dict,
    rshape: tuple[int, int],
) -> tuple[slice, slice]:
    """Scan-space ROI → half-open slices ``(y, x)``.

    ``bounds`` uses the same keys as the other notebooks:
    ``xMin, xMax, yMin, yMax`` (rx, ry on the virtual image).
    """
    Ry, Rx = int(rshape[0]), int(rshape[1])
    x0 = int(np.floor(float(bounds["xMin"])))
    x1 = int(np.ceil(float(bounds["xMax"])))
    y0 = int(np.floor(float(bounds["yMin"])))
    y1 = int(np.ceil(float(bounds["yMax"])))
    y0 = max(0, min(Ry, y0))
    y1 = max(0, min(Ry, y1))
    x0 = max(0, min(Rx, x0))
    x1 = max(0, min(Rx, x1))
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"empty ROI after clip: y[{y0}:{y1}] x[{x0}:{x1}]")
    return slice(y0, y1), slice(x0, x1)


def _cube4d(cube) -> np.ndarray:
    data = getattr(cube, "data", cube)
    arr = data if isinstance(data, np.ndarray) else np.asarray(data)
    if arr.ndim != 4:
        raise ValueError(f"expected (Ry, Rx, Qy, Qx), got shape {arr.shape}")
    return arr


def mean_dp_from_4d(cube, slices: tuple[slice, slice]) -> np.ndarray:
    """Average patterns in the ROI. Output shape is the detector ``(Qy, Qx)``."""
    arr = _cube4d(cube)
    sy, sx = slices
    block = arr[sy, sx]
    if block.size == 0:
        raise ValueError("empty ROI")
    return np.asarray(block, dtype=np.float64).mean(axis=(0, 1))


def n_patterns_in_slices(slices: tuple[slice, slice]) -> int:
    sy, sx = slices
    if sy.start is None or sy.stop is None or sx.start is None or sx.stop is None:
        raise ValueError("ROI slices must be bounded (start and stop)")
    ny = int(sy.stop) - int(sy.start)
    nx = int(sx.stop) - int(sx.start)
    return int(max(0, ny) * max(0, nx))


def prepare_dp_array(dp, *, dtype: str = "uint16") -> np.ndarray:
    """Same ``(Qy, Qx)`` — never resample. ``uint16`` is what CrystBox likes."""
    a = np.asarray(dp, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"DP must be 2D, got {a.shape}")
    kind = str(dtype).lower()
    if kind in ("float32", "f4"):
        return a.astype(np.float32, copy=False)
    if kind in ("uint16", "u2"):
        pos = np.clip(a, 0.0, None)
        mx = float(pos.max()) if pos.size else 0.0
        if mx <= 0.0:
            return np.zeros(a.shape, dtype=np.uint16)
        return np.round(pos * (65535.0 / mx)).astype(np.uint16)
    raise ValueError(f"unsupported TIFF dtype: {dtype}")


def write_dp_tiff(
    path: str | Path,
    dp,
    *,
    q_Ainv_per_px: float | None = None,
    q_nminv_per_px: float | None = None,
    roi: dict | None = None,
    n_patterns: int | None = None,
    source: str | None = None,
    dtype: str = "uint16",
) -> dict[str, Any]:
    """Write a 2D TIFF at native detector pixels + a JSON sidecar for CrystBox."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(dp)
    if raw.ndim != 2:
        raise ValueError(f"refusing to write a resampled/ND array: {raw.shape}")
    out = prepare_dp_array(raw, dtype=dtype)
    if out.shape != raw.shape:
        raise RuntimeError("internal: TIFF array shape changed")

    if q_nminv_per_px is None and q_Ainv_per_px is not None:
        q_nminv_per_px = q_Ainv_to_nminv(q_Ainv_per_px)

    _imwrite(path, out)

    meta: dict[str, Any] = {
        "file": str(path.resolve()),
        "shape": [int(out.shape[0]), int(out.shape[1])],
        "tiff_dtype": str(out.dtype),
        "source_min": float(np.min(raw)) if raw.size else None,
        "source_max": float(np.max(raw)) if raw.size else None,
        "q_Ainv_per_px": float(q_Ainv_per_px) if q_Ainv_per_px is not None else None,
        "q_nminv_per_px": float(q_nminv_per_px) if q_nminv_per_px is not None else None,
        "q_units_for_crystbox": "1/nm / px",
        "resampled": False,
        "roi": roi,
        "n_patterns": int(n_patterns) if n_patterns is not None else None,
        "source": source,
    }
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _imwrite(path: Path, arr: np.ndarray) -> None:
    """Prefer tifffile (standard grayscale TIFF). PIL fallback keeps native pixels."""
    try:
        import tifffile
        tifffile.imwrite(path, arr, photometric="minisblack")
        return
    except ImportError:
        import warnings
        warnings.warn(
            "tifffile not installed — writing TIFF with Pillow (pixels unchanged; "
            "CrystBox tags may differ). pip install tifffile",
            stacklevel=2,
        )
    from PIL import Image
    Image.fromarray(arr).save(path)


_DP_LABELS = {"dp_mean": "DP_mean", "dp_max": "DP_max"}
_SIDECAR_SKIP = ("braggpeaks",)


def virtual_dp_stem(path: str | Path) -> str:
    stem = Path(path).stem
    low = stem.lower()
    if low.endswith("_braggpeaks"):
        return stem[: -len("_braggpeaks")]
    return stem


def virtual_dp_export_name(path_or_stem: str | Path, kind: str) -> str:
    """``Scan01.h5`` + ``dp_max`` → ``Scan01_DP_max.tif``."""
    label = _DP_LABELS.get(str(kind), str(kind))
    return f"{virtual_dp_stem(path_or_stem)}_{label}.tif"


def collect_sidecar_h5(
    folder: str | Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Companion ``ScanXX.h5`` files — skip ``*braggpeaks*.h5``."""
    root = Path(folder)
    it = root.rglob("*.h5") if recursive else root.glob("*.h5")
    out: list[Path] = []
    for p in sorted(it):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(s in name for s in _SIDECAR_SKIP):
            continue
        out.append(p.resolve())
    return out


def read_sidecar_dps(path: str | Path) -> dict[str, np.ndarray]:
    """Read ``dp_mean`` / ``dp_max`` from a Fast4D sidecar. Does not open the 4D cube."""
    import h5py

    wanted = set(_DP_LABELS)
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            base = name.split("/")[-1]
            if base not in wanted or base in out:
                return
            ds = None
            if isinstance(obj, h5py.Group) and "data" in obj:
                child = obj["data"]
                if getattr(child, "ndim", None) == 2:
                    ds = child
            elif isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                ds = obj
            if ds is None:
                return
            out[base] = np.asarray(ds, dtype=np.float64)

        f.visititems(visit)
    return out


def export_sidecar_dps(
    h5_path: str | Path,
    out_dir: str | Path,
    *,
    q_Ainv_per_px: float | None = None,
    keys: tuple[str, ...] = ("dp_mean", "dp_max"),
    dtype: str = "uint16",
) -> list[Path]:
    """Write ``ScanXX_DP_mean.tif`` and ``ScanXX_DP_max.tif`` (native Q pixels)."""
    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dps = read_sidecar_dps(h5_path)
    written: list[Path] = []
    for key in keys:
        arr = dps.get(key)
        if arr is None:
            continue
        dest = out_dir / virtual_dp_export_name(h5_path, key)
        write_dp_tiff(
            dest, arr,
            q_Ainv_per_px=q_Ainv_per_px,
            source=str(h5_path.resolve()),
            dtype=dtype,
        )
        written.append(dest)
    return written


def export_all_sidecar_dps(
    folder: str | Path,
    out_dir: str | Path | None = None,
    *,
    recursive: bool = False,
    q_Ainv_per_px: float | None = None,
    dtype: str = "uint16",
) -> list[Path]:
    """Export DP mean/max TIFFs for every sidecar under ``folder``."""
    dest = Path(out_dir) if out_dir is not None else Path(folder) / "DP_exports"
    written: list[Path] = []
    seen: set[str] = set()
    for h5 in collect_sidecar_h5(folder, recursive=recursive):
        for key in ("dp_mean", "dp_max"):
            name = virtual_dp_export_name(h5, key)
            if name in seen:
                raise ValueError(f"TIFF name collision: {name} (from {h5})")
            seen.add(name)
        written.extend(
            export_sidecar_dps(h5, dest, q_Ainv_per_px=q_Ainv_per_px, dtype=dtype)
        )
    return written
