from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

try:
    from .state import WorkflowState
except ImportError:
    from state import WorkflowState


VIRTUAL_IMAGE_KEYS = (
    "annular_dark_field",
    "bright_field",
    "dp_mean",
    "dp_max",
)

BRAGG_SUMMARY_ATTRS = (
    "Rshape",
    "Qshape",
    "shape",
    "raw",
    "pointlists",
    "get_pointlist",
    "data",
    "calstate",
    "histogram",
)


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)


def cupy_device_summary() -> str:
    """Short CuPy/CUDA status string for GUI hints (virtualization vs disk detection)."""
    try:
        import cupy as cp
    except ImportError:
        return "CuPy: not installed — GPU disk detection (Steps 7–8) will not run on CUDA."
    try:
        n = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return f"CuPy: could not query devices ({exc})"
    if n <= 0:
        return "CuPy: 0 CUDA devices visible — check drivers / CUDA toolkit."
    try:
        dev_id = int(cp.cuda.Device().id)
        props = cp.cuda.runtime.getDeviceProperties(dev_id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        free_b, total_b = cp.cuda.Device(dev_id).mem_info
        return (
            f"CuPy: {n} CUDA device(s) | using GPU {dev_id}: {name} | "
            f"VRAM free {free_b / (1024**3):.1f} / {total_b / (1024**3):.1f} GiB"
        )
    except Exception as exc:
        return f"CuPy: {n} device(s) reported, but query failed ({exc})"


# py4DSTEM ``get_maxima_2D`` / ``find_Bragg_disks`` only accept these strings
# (see ``preprocess.utils.get_maxima_2D``: ``('pixel', 'poly', 'multicorr')``).
_SUBPIXEL_PY4DSTEM = frozenset({"pixel", "poly", "multicorr"})


def normalize_subpixel_keyword(value: Any) -> str:
    """
    Map GUI labels / legacy saved params to a valid ``subpixel=`` for
    ``DataCube.find_Bragg_disks``.

    - ``pixel`` — integer-pixel peaks (older notebooks / GUI called this ``none``).
    - ``poly`` — polynomial refinement.
    - ``multicorr`` — correlation / COM-style refinement (GUI historically used ``com``).
    """
    s = str(value or "").strip().lower()
    if s in _SUBPIXEL_PY4DSTEM:
        return s
    if s in ("none", "no", "", "int", "integer"):
        return "pixel"
    if s in ("com", "corr", "correlation", "crosscorr", "cross_corr", "multicorrelation"):
        return "multicorr"
    return "poly"


def _ensure_cupy_current_device_for_thread(
    detect_params: dict[str, Any],
    log: Callable[[str], None] | None = None,
) -> None:
    """
    CuPy binds the "current" CUDA device to each OS thread. In Jupyter, find_Bragg_disks usually
    runs on the main thread; this app runs it in a background thread, so without an explicit
    cuda.Device().use() here nvidia-smi can stay flat even with CUDA=True in detect_params.
    """
    if not (bool(detect_params.get("CUDA")) or bool(detect_params.get("CUDA_batched"))):
        return
    try:
        import cupy as cp
    except ImportError:
        _log(log, "CuPy not importable — find_Bragg_disks cannot use the GPU in this environment.")
        return
    try:
        raw_dev = detect_params.get("CUDA_device", detect_params.get("cuda_device", 0))
        dev_id = int(raw_dev)
    except Exception:
        dev_id = 0
    try:
        cp.cuda.Device(dev_id).use()
        _log(log, f"CuPy: cuda.Device({dev_id}).use() in this thread (GPU disk detection).")
    except Exception as exc:
        _log(log, f"CuPy: cuda.Device({dev_id}).use() failed ({exc}) — GPU may not be used.")


def _shape_message(name: str, obj: Any) -> str:
    return f"{name} Rshape: {getattr(obj, 'Rshape', None)} | Qshape: {getattr(obj, 'Qshape', None)}"


def summarize_obj(obj: Any) -> list[str]:
    return [attr for attr in BRAGG_SUMMARY_ATTRS if hasattr(obj, attr)]


# Raw scan files Step 1 / Step 2 can load (non-.h5 sidecar is separate).
# Keep in sync with `_load_raw_datacube` and GUI file dialogs.
SUPPORTED_RAW_SCAN_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mib",
        ".npy",
        ".npz",
        ".h5",
        ".hdf5",
        ".dm3",
        ".dm4",
        ".gtg",
    }
)


def raw_data_dialog_filetypes() -> list[tuple[str, str]]:
    """
    Filedialog patterns for tkinter (Windows: use ``*.a;*.b`` in the pattern string).
    """
    all_supported = ";".join(f"*{ext}" for ext in sorted(SUPPORTED_RAW_SCAN_EXTENSIONS))
    return [
        ("All supported 4D-STEM", all_supported),
        ("Merlin / MIB", "*.mib"),
        ("NumPy / NPZ", "*.npy;*.npz"),
        ("py4DSTEM / EMD (.h5)", "*.h5;*.hdf5"),
        ("DigitalMicrograph", "*.dm3;*.dm4"),
        ("Gatan K2 (.gtg)", "*.gtg"),
        ("All files", "*.*"),
    ]


def _default_sidecar_h5_path(raw_path: Path) -> Path:
    """
    Default path for optional precomputed virtual-image .h5.

    When the raw scan is already an EMD/HDF5 file, a sibling ``*_precomputed.h5`` is tried first
    so we do not assume the raw file itself contains the virtualization tree.
    """
    if raw_path.suffix.lower() in (".h5", ".hdf5"):
        sibling = raw_path.with_name(f"{raw_path.stem}_precomputed.h5")
        if sibling.exists():
            return sibling
        return raw_path
    return raw_path.with_suffix(".h5")


def _normalize_import_file_mem(mem: str | None) -> str:
    """Map GUI / MIB mem keywords to ``py4DSTEM.import_file`` (RAM | MEMMAP)."""
    if mem in (None, "", "auto"):
        return "MEMMAP"
    mlow = str(mem).strip().lower()
    if mlow == "ram":
        return "RAM"
    return "MEMMAP"


def _coerce_loaded_to_datacube(obj: Any, *, source: str):
    """Turn ``py4DSTEM.read`` / ``import_file`` results into a DataCube-like object."""
    if obj is None:
        raise ValueError(f"Load returned None ({source})")

    if hasattr(obj, "Rshape") and hasattr(obj, "Qshape"):
        return obj

    data = getattr(obj, "data", None)
    if isinstance(data, np.ndarray) and data.ndim == 4:
        return _wrap_numpy_as_datacube(data)

    tree = getattr(obj, "tree", None)
    if isinstance(tree, dict):
        for v in tree.values():
            try:
                return _coerce_loaded_to_datacube(v, source=source)
            except ValueError:
                continue

    for name in ("datacube", "root", "_root"):
        child = getattr(obj, name, None)
        if child is not None and child is not obj:
            try:
                return _coerce_loaded_to_datacube(child, source=source)
            except ValueError:
                continue

    arr_try = np.asarray(obj)
    if arr_try.ndim == 4:
        return _wrap_numpy_as_datacube(arr_try)

    raise ValueError(
        f"Could not obtain a 4D DataCube from {source} (type {type(obj)!r}). "
        "For EMD/HDF5, try py4DSTEM.print_h5_tree(filepath) to pick the correct group path."
    )


def _load_emd_h5_datacube(
    path: Path,
    *,
    emd_datapath: str | None = None,
    log: Callable[[str], None] | None = None,
):
    import py4DSTEM

    paths_to_try: list[str | None] = []
    if emd_datapath is not None:
        paths_to_try.append(emd_datapath)
    paths_to_try.append(None)
    paths_to_try.append("datacube_root")
    for extra in _candidate_datapaths(path):
        if extra not in paths_to_try:
            paths_to_try.append(extra)

    last_exc: Exception | None = None
    for dp in paths_to_try:
        try:
            out = py4DSTEM.read(filepath=str(path), datapath=dp, tree=True)
            if isinstance(out, list):
                _log(log, f"py4DSTEM.read: multiple EMD roots for datapath={dp!r}; trying each.")
                for item in out:
                    try:
                        return _coerce_loaded_to_datacube(item, source=str(path))
                    except ValueError:
                        continue
                continue
            return _coerce_loaded_to_datacube(out, source=str(path))
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(f"Failed to read EMD/HDF5 as a 4D DataCube: {path}. Last error: {last_exc}") from last_exc


def _load_npz_datacube(path: Path):
    z = np.load(str(path), allow_pickle=True)
    try:
        best = None
        best_size = -1
        for k in z.files:
            a = z[k]
            if getattr(a, "ndim", 0) == 4 and int(a.size) > best_size:
                best = a
                best_size = int(a.size)
        if best is None:
            raise ValueError(f"No 4D array found in {path} (keys={list(z.files)})")
        return _wrap_numpy_as_datacube(best)
    finally:
        try:
            z.close()
        except Exception:
            pass


def _extension_import_filetype(path: Path) -> str | None:
    """Optional ``filetype=`` hint for ``py4DSTEM.import_file`` (None = auto)."""
    ext = path.suffix.lower()
    if ext in (".dm3", ".dm4"):
        return "dm"
    if ext == ".gtg":
        return "gatan_K2_bin"
    return None


def _load_via_import_file(
    path: Path,
    *,
    mem: str | None = None,
    binfactor: int = 1,
    scan: tuple[int, int] | None = None,
    filetype: str | None = None,
):
    import py4DSTEM

    mem_imp = _normalize_import_file_mem(mem)
    kwargs: dict[str, Any] = {"mem": mem_imp, "binfactor": max(1, int(binfactor))}
    if filetype is not None:
        kwargs["filetype"] = filetype
    if scan is not None:
        kwargs["scan"] = scan

    def _call(kw: dict[str, Any]) -> Any:
        return py4DSTEM.import_file(str(path), **kw)

    try:
        out = _call(kwargs)
    except TypeError:
        kwargs.pop("scan", None)
        try:
            out = _call(kwargs)
        except TypeError:
            kwargs.pop("filetype", None)
            out = _call(kwargs)

    return _coerce_loaded_to_datacube(out, source=str(path))


def _pick_mib_mem_mode(path: Path) -> str | None:
    """
    Choose a safe default for `read_mib.load_mib(mem=...)`.

    Large 4D cubes (e.g. 512x512x256x256) are ~32GB as float32; forcing RAM is often
    impossible. An explicit FAST4D_FORCE_MEMMAP=1 environment variable always wins,
    for memory-constrained machines or users who know their datasets are large.
    """
    import os

    if os.environ.get("FAST4D_FORCE_MEMMAP", "").strip() == "1":
        return "memmap"

    try:
        size = int(path.stat().st_size)
    except Exception:
        size = 0

    # Heuristic: if the file is at least moderately large, prefer memmap even before
    # we know the final shape. Lowered from 6 GiB to 2 GiB (2026-07-09 memory report):
    # most workstation RAM budgets can't afford more than a couple of RAM-resident
    # cubes at 2+ GiB anyway, and memmap's cost is amortized I/O, not correctness risk.
    if size >= 2 * 1024**3:  # >= ~2 GiB on disk
        return "memmap"

    try:
        import psutil  # type: ignore

        avail = int(psutil.virtual_memory().available)
        # If the file is > ~25% of available RAM, don't default to RAM (was 40%).
        if size > 0 and size > int(0.25 * max(avail, 1)):
            return "memmap"
    except Exception:
        pass

    return None


def _mib_scan_from_reshape_mismatch(exc: Exception) -> tuple[int, int] | None:
    """
    py4DSTEM ``read_mib`` sometimes assumes a square scan grid (e.g. 256×256) when the Merlin MIB
    encodes a larger scan (e.g. 512×512) with the same Q shape. That yields::

        ValueError: cannot reshape array of size S into shape (Ry, Rx, Qy, Qx)

    If Qy,Qx in the failing shape are correct, real-space pixels are ``S / (Qy*Qx)`` and the
    assumed ``Ry*Rx`` is often short by a perfect-square factor (equal scale on both axes).
    """
    msg = str(exc)
    m = re.search(r"cannot reshape array of size (\d+) into shape \(([\d, ]+)\)", msg)
    if not m:
        return None
    try:
        total = int(m.group(1))
        dims = [int(x.strip()) for x in m.group(2).split(",") if x.strip()]
    except ValueError:
        return None
    if len(dims) != 4:
        return None
    ry_w, rx_w, qy, qx = dims[0], dims[1], dims[2], dims[3]
    q_prod = qy * qx
    if q_prod <= 0 or total % q_prod != 0:
        return None
    scan_prod = total // q_prod
    base = ry_w * rx_w
    if base <= 0 or scan_prod % base != 0:
        return None
    factor = scan_prod // base
    if factor <= 1:
        return None
    root = int(round(factor**0.5))
    if root * root != factor:
        return None
    return (ry_w * root, rx_w * root)


def _mib_truncation_diagnosis(exc: Exception) -> str | None:
    """
    Diagnose the common case where a Merlin acquisition was stopped early: the file holds fewer
    frames than the scan grid in the reshape target declares (e.g. .hdr promises 256x256 but the
    .mib was cut off after ~15 lines). Returns a human-readable explanation, or None if the
    failure doesn't look like a truncated acquisition (e.g. it's the larger-scan case already
    handled by ``_mib_scan_from_reshape_mismatch``).
    """
    msg = str(exc)
    m = re.search(r"cannot reshape array of size (\d+) into shape \(([\d, ]+)\)", msg)
    if not m:
        return None
    try:
        total = int(m.group(1))
        dims = [int(x.strip()) for x in m.group(2).split(",") if x.strip()]
    except ValueError:
        return None
    if len(dims) != 4:
        return None
    ry_w, rx_w, qy, qx = dims
    q_prod = qy * qx
    declared_frames = ry_w * rx_w
    if q_prod <= 0 or declared_frames <= 0:
        return None
    actual_frames = total // q_prod
    remainder_px = total % q_prod
    if actual_frames >= declared_frames:
        return None
    full_rows = actual_frames // rx_w if rx_w > 0 else 0
    leftover = actual_frames % rx_w if rx_w > 0 else actual_frames
    partial_note = (
        f", plus a partial frame ({remainder_px} of {q_prod} pixels)" if remainder_px else ""
    )
    return (
        f"this .mib looks like an interrupted/incomplete acquisition: it holds {actual_frames} "
        f"of the {declared_frames} frames a {ry_w}x{rx_w} scan needs "
        f"({full_rows} full scan line(s) + {leftover} extra frame(s){partial_note}). "
        f"Re-acquire the scan or point Fast4D at the complete .mib file."
    )


def _load_mib(
    path: Path,
    *,
    mem: str | None = None,
    binfactor: int = 1,
    scan: tuple[int, int] | None = None,
    log: Callable[[str], None] | None = None,
):
    """
    Load a MIB into a py4DSTEM DataCube.

    `binfactor` > 1 bins diffraction space on load (memory and compute scale ~ 1/binfactor**2 in Q).
    `scan` is (Ry, Rx) when MIB metadata does not encode the real-space grid (see py4DSTEM read_mib docs).
    """
    from py4DSTEM.io.filereaders import read_mib

    chosen = mem if mem not in (None, "", "auto") else _pick_mib_mem_mode(path)
    bf = max(1, int(binfactor))
    extra_primary: dict = {}
    if bf != 1:
        extra_primary["binfactor"] = bf
    if scan is not None:
        extra_primary["scan"] = scan

    def _try_one(mode: str | None, with_extras: bool, extras: dict) -> Any:
        kwargs: dict = {}
        if mode is not None:
            kwargs["mem"] = mode
        if with_extras and extras:
            kwargs.update(extras)
        if not kwargs:
            return read_mib.load_mib(str(path))
        try:
            return read_mib.load_mib(str(path), **kwargs)
        except TypeError:
            # Older py4DSTEM: drop extras, then drop mem=.
            if with_extras and extras:
                return _try_one(mode, False, {})
            if mode is not None and "mem" in kwargs:
                return read_mib.load_mib(str(path))
            raise

    # Try requested mode first, then fall back in a sensible order.
    attempts: list[str | None] = []
    if chosen is None:
        attempts = [None, "memmap", "RAM"]
    else:
        attempts = [chosen]
        for fallback in ("memmap", None, "RAM"):
            if fallback not in attempts:
                attempts.append(fallback)

    last_exc: Exception | None = None
    tried_inferred: tuple[int, int] | None = None
    for mode in attempts:
        try:
            return _try_one(mode, True, extra_primary)
        except Exception as exc:
            last_exc = exc
            continue

    if scan is None and last_exc is not None:
        inferred = _mib_scan_from_reshape_mismatch(last_exc)
        if inferred is not None:
            tried_inferred = inferred
            _log(
                log,
                f"MIB grid metadata looks inconsistent with file size; retrying read_mib with "
                f"scan={inferred} (inferred from reshape error).",
            )
            extra_inferred = dict(extra_primary)
            extra_inferred["scan"] = inferred
            for mode in attempts:
                try:
                    return _try_one(mode, True, extra_inferred)
                except Exception as exc:
                    last_exc = exc
                    continue

    hint = ""
    if tried_inferred is not None:
        hint = f" (also tried inferred scan={tried_inferred})"
    diagnosis = _mib_truncation_diagnosis(last_exc) if last_exc is not None else None
    if diagnosis is not None:
        raise RuntimeError(
            f"Failed to load MIB: {path} — {diagnosis} "
            f"(tried modes={attempts}, binfactor={bf}, scan={scan}){hint}. Raw error: {last_exc}"
        )
    raise RuntimeError(
        f"Failed to load MIB: {path} (tried modes={attempts}, binfactor={bf}, scan={scan}){hint}. "
        f"Last error: {last_exc}"
    )


def _wrap_numpy_as_datacube(array_ryxqyqx: np.ndarray):
    """
    Wrap a numpy array shaped (Ry, Rx, Qy, Qx) as a py4DSTEM DataCube.

    This enables running the same workflow on simulated/synthetic cubes stored as .npy.
    """
    import py4DSTEM

    DataCube = getattr(py4DSTEM, "DataCube", None)
    if DataCube is None:
        try:
            from py4DSTEM.datacube import DataCube as _DataCube  # type: ignore
        except Exception as exc:
            raise RuntimeError("Could not import py4DSTEM.DataCube.") from exc
        DataCube = _DataCube

    arr = np.asarray(array_ryxqyqx)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array shaped (Ry,Rx,Qy,Qx), got {arr.shape}")

    # Try the most common constructor patterns across versions.
    try:
        return DataCube(arr)
    except TypeError:
        return DataCube(data=arr)


def _load_raw_datacube(
    path: Path,
    *,
    mem: str | None = None,
    binfactor: int = 1,
    scan: tuple[int, int] | None = None,
    emd_datapath: str | None = None,
    log: Callable[[str], None] | None = None,
):
    """
    Load a 4D scan into a py4DSTEM ``DataCube``.

    Supported:

    - ``.mib`` — Merlin (``read_mib``, with mem/binfactor/scan heuristics).
    - ``.npy`` — NumPy cube (Ry, Rx, Qy, Qx); memory-mapped read.
    - ``.npz`` — first/largest 4D array among keys.
    - ``.h5`` / ``.hdf5`` — py4DSTEM / EMD 1.0 (``py4DSTEM.read``).
    - ``.dm3`` / ``.dm4``, ``.gtg``, EMPAD, Arina, abTEM exports, etc. — ``py4DSTEM.import_file``
      (detection + optional extension hints).

    See also: ``SUPPORTED_RAW_SCAN_EXTENSIONS``, ``raw_data_dialog_filetypes``.
    """
    suf = path.suffix.lower()
    if suf == ".mib":
        return _load_mib(path, mem=mem, binfactor=binfactor, scan=scan, log=log)
    if suf == ".npy":
        arr = np.load(str(path), mmap_mode="r")
        return _wrap_numpy_as_datacube(arr)
    if suf == ".npz":
        return _load_npz_datacube(path)
    if suf in (".h5", ".hdf5"):
        return _load_emd_h5_datacube(path, emd_datapath=emd_datapath, log=log)

    hint = _extension_import_filetype(path)
    return _load_via_import_file(path, mem=mem, binfactor=binfactor, scan=scan, filetype=hint)


def _load_datacube_root(h5_path: Path):
    import py4DSTEM

    return py4DSTEM.read(filepath=str(h5_path), datapath="datacube_root")


def _candidate_datapaths(h5_path: Path) -> list[str | None]:
    """Return possible py4DSTEM datapaths without changing the HDF5 file."""

    candidates: list[str | None] = ["datacube_root"]
    try:
        import h5py

        with h5py.File(h5_path, "r") as h5:
            def visitor(name: str, obj) -> None:
                if name.split("/")[-1] == "datacube_root":
                    candidates.append(name)

            h5.visititems(visitor)
    except Exception:
        pass

    candidates.append(None)

    unique: list[str | None] = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    return unique


def _has_required_virtual_images(obj: Any) -> bool:
    for key in VIRTUAL_IMAGE_KEYS:
        try:
            _read_tree(obj, key)
        except Exception:
            return False
    return True


def _load_visualcube_from_h5(h5_path: Path, log: Callable[[str], None] | None = None):
    """Load the object that owns the precomputed virtual-image tree."""

    import py4DSTEM

    errors: list[str] = []
    for datapath in _candidate_datapaths(h5_path):
        label = "<root>" if datapath is None else datapath
        try:
            if datapath is None:
                obj = py4DSTEM.read(filepath=str(h5_path))
            else:
                obj = py4DSTEM.read(filepath=str(h5_path), datapath=datapath)
        except Exception as exc:
            errors.append(f"{label}: read failed ({exc})")
            continue

        if _has_required_virtual_images(obj):
            _log(log, f"Found precomputed virtual-image tree at datapath: {label}")
            return obj
        errors.append(f"{label}: readable but missing one or more virtual images")

    raise RuntimeError("No readable datapath with all required virtual images. Tried: " + " | ".join(errors))


def _read_tree(obj: Any, key: str):
    """Read py4DSTEM tree entries from DataCube or Root-like objects."""

    if obj is None or not hasattr(obj, "tree"):
        raise TypeError("Loaded .h5 object does not provide tree().")

    try:
        return obj.tree(key)
    except Exception as direct_error:
        try:
            root = obj.tree("datacube_root")
            return root.tree(key)
        except Exception as root_error:
            raise RuntimeError(
                f"Could not read '{key}' directly or under 'datacube_root': "
                f"{direct_error}; {root_error}"
            ) from root_error


def make_central_vacuum_roi(rshape, fraction: float = 0.05) -> np.ndarray:
    """Create the notebook central ROI: center +/- 5 percent of scan size."""

    height = int(rshape[0])
    width = int(rshape[1]) if len(rshape) > 1 else height
    cy = int(round(height / 2))
    cx = int(round(width / 2))
    hy = int(round(float(fraction) * height))
    hx = int(round(float(fraction) * width))

    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, cy - hy):min(height, cy + hy), max(0, cx - hx):min(width, cx + hx)] = True
    return mask


def load_data_step(
    state: WorkflowState,
    raw_mib_path: str | Path,
    precomputed_h5_path: str | Path | None = None,
    braggpeaks_path: str | Path | None = None,
    use_existing_braggpeaks: bool = False,
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Step 1: load raw 4D-STEM data and optional precomputed datacube_root."""

    raw_path = Path(raw_mib_path).expanduser()
    h5_path = Path(precomputed_h5_path).expanduser() if precomputed_h5_path else _default_sidecar_h5_path(raw_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file does not exist: {raw_path}")

    low = raw_path.suffix.lower()
    if low not in SUPPORTED_RAW_SCAN_EXTENSIONS:
        _log(
            log,
            f"Note: extension {raw_path.suffix!r} is not in the curated list {sorted(SUPPORTED_RAW_SCAN_EXTENSIONS)}; "
            "trying py4DSTEM.import_file anyway.",
        )

    same_paths = state.raw_mib_path == raw_path and state.precomputed_h5_path == h5_path
    if same_paths and state.datacube is not None:
        _log(log, "Data already loaded for these paths; reusing current datacube.")
        return state

    state.reset_data_products()
    state.reset_probe_products()
    state.raw_mib_path = raw_path
    state.precomputed_h5_path = h5_path

    if h5_path.exists():
        try:
            state.visualcube = _load_visualcube_from_h5(h5_path, log=log)
            _log(log, f"Loaded valid precomputed .h5: {h5_path}")
            _log(log, _shape_message("visualcube", state.visualcube))
        except Exception as exc:
            state.visualcube = None
            _log(log, f"Warning: .h5 exists but no valid virtual-image tree was loaded: {exc}")
            _log(log, "Fallback: loading raw scan file.")
    else:
        _log(log, f"Optional .h5 not found: {h5_path}")
        _log(log, "Fallback: loading raw scan file.")

    state.datacube = _load_raw_datacube(raw_path, log=log)
    _log(log, f"Loaded raw data: {raw_path}")
    _log(log, _shape_message("datacube", state.datacube))

    state.recommended_q_pixel_A_inv_per_px = None
    meta_path = raw_path.with_name(f"{raw_path.stem}_meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            qpx = meta.get("q_pixel_size_A_inv_per_px")
            if qpx is not None:
                state.recommended_q_pixel_A_inv_per_px = float(qpx)
                _log(
                    log,
                    f"Simulator metadata ({meta_path.name}): recommended Q pixel size = "
                    f"{float(qpx):.10g} A^-1/px — Step 11 slider can use this; re-run Step 8 if braggpeaks "
                    "were computed with a different calibration.",
                )
        except Exception as exc:
            _log(log, f"Note: could not read simulator meta {meta_path}: {exc}")

    state.use_existing_braggpeaks = bool(use_existing_braggpeaks)
    if use_existing_braggpeaks:
        if braggpeaks_path is None or str(braggpeaks_path).strip() == "":
            default_bragg = raw_path.with_name(f"{raw_path.stem}braggpeaks.h5")
            bragg_path = default_bragg
        else:
            bragg_path = Path(braggpeaks_path).expanduser()
        state.braggpeaks_path = bragg_path
        state.braggpeaks = load_braggpeaks_file(bragg_path, expected_rshape=state.datacube.Rshape, log=log)
        _log(log, "Existing braggpeaks loaded; point selection, detection preview, and full computation can be skipped.")
    return state


def compute_probe_step(
    state: WorkflowState,
    vacuum_mib_path: str | Path,
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Step 2: load vacuum, compute probe, extract probe params, build sigmoid kernel."""

    vacuum_path = Path(vacuum_mib_path).expanduser()
    if not vacuum_path.exists():
        raise FileNotFoundError(f"Vacuum file does not exist: {vacuum_path}")

    if state.vacuum_mib_path == vacuum_path and state.probe is not None:
        _log(log, "Probe already computed for this vacuum path; reusing current probe.")
        return state

    state.reset_probe_products()
    state.vacuum_mib_path = vacuum_path

    state.vacuumcube = _load_raw_datacube(vacuum_path, log=log)
    _log(log, f"Loaded vacuum data: {vacuum_path}")
    _log(log, _shape_message("vacuumcube", state.vacuumcube))

    roi = make_central_vacuum_roi(state.vacuumcube.Rshape, fraction=0.05)

    def _probe_workflow():
        pr = state.vacuumcube.get_vacuum_probe(ROI=roi)
        alpha_pr, qx0_pr, qy0_pr = state.vacuumcube.get_probe_size(pr.probe)
        pr.get_kernel(mode="sigmoid", radii=(alpha_pr, 2 * alpha_pr))
        return pr, alpha_pr, qx0_pr, qy0_pr

    probe, alpha_pr, qx0_pr, qy0_pr = _with_plot_suppressed(_probe_workflow)

    state.probe = probe
    state.probe_alpha = float(alpha_pr)
    state.probe_qx0 = float(qx0_pr)
    state.probe_qy0 = float(qy0_pr)

    state.probe_source = "vacuum_mib"

    _log(
        log,
        "Computed vacuum probe and sigmoid kernel: "
        f"alpha_pr={state.probe_alpha:.6g}, "
        f"qx0_pr={state.probe_qx0:.6g}, "
        f"qy0_pr={state.probe_qy0:.6g}",
    )
    return state


def set_probe_bf_vacuum_roi_from_bounds(
    state: WorkflowState,
    bounds: tuple[int, int, int, int],
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Store real-space ROI mask (Rshape) for ``datacube.get_vacuum_probe(ROI=...)`` on the main datacube."""

    if state.datacube is None:
        raise RuntimeError("Load datacube before setting probe vacuum ROI.")
    x0, x1, y0, y1 = map(int, bounds)
    height, width = map(int, state.datacube.Rshape)
    x0 = max(0, min(x0, width - 1))
    x1 = max(1, min(x1, width))
    y0 = max(0, min(y0, height - 1))
    y1 = max(1, min(y1, height))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Invalid probe ROI bounds; require x0 < x1 and y0 < y1.")
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    state.probe_bf_roi_bounds = (x0, x1, y0, y1)
    state.probe_bf_roi_mask = mask
    # ROI changed: any existing probe is no longer valid until user recomputes.
    state.probe = None
    state.probe_source = None
    state.probe_alpha = None
    state.probe_qx0 = None
    state.probe_qy0 = None
    _log(
        log,
        f"Probe vacuum ROI (use bare vacuum on ADF or BF): x[{x0}:{x1}] y[{y0}:{y1}] "
        f"pixels={int(mask.sum())}",
    )
    return state


def compute_probe_from_bf_vacuum_roi_step(
    state: WorkflowState,
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """
    Notebook ``basics_02_diskdetection``: average DPs over a user mask on the main datacube, then ``get_kernel``.

    The resulting ``state.probe`` (with ``.kernel``) is the same object type used by
    ``find_Bragg_disks(template=<2D ndarray from probe.kernel>, **detect_params)`` in Steps 7–8.
    """

    if state.datacube is None:
        raise RuntimeError("Load datacube before computing probe from BF ROI.")
    if state.probe_bf_roi_mask is None or state.probe_bf_roi_bounds is None:
        raise RuntimeError("Define the vacuum ROI on the main-scan virtual image first (Step 2).")

    roi = np.asarray(state.probe_bf_roi_mask, dtype=bool)
    bounds = state.probe_bf_roi_bounds
    state.reset_probe_products()
    state.probe_bf_roi_bounds = bounds
    state.probe_bf_roi_mask = roi

    if roi.shape != tuple(int(x) for x in state.datacube.Rshape):
        raise RuntimeError(f"Probe ROI mask shape {roi.shape} != datacube Rshape {state.datacube.Rshape}.")
    if not bool(np.any(roi)):
        raise RuntimeError("Probe vacuum ROI mask is empty.")

    _log(log, f"get_vacuum_probe(ROI=mask) on main datacube | Rshape={state.datacube.Rshape} | ROI pixels={int(roi.sum())}")

    def _probe_workflow():
        pr = state.datacube.get_vacuum_probe(ROI=roi)
        alpha_pr, qx0_pr, qy0_pr = state.datacube.get_probe_size(pr.probe)
        pr.get_kernel(mode="sigmoid", radii=(alpha_pr, 2 * alpha_pr))
        return pr, alpha_pr, qx0_pr, qy0_pr

    probe, alpha_pr, qx0_pr, qy0_pr = _with_plot_suppressed(_probe_workflow)

    state.probe = probe
    state.probe_source = "bf_roi"
    state.probe_alpha = float(alpha_pr)
    state.probe_qx0 = float(qx0_pr)
    state.probe_qy0 = float(qy0_pr)
    state.vacuum_mib_path = None

    _log(
        log,
        "Probe from BF vacuum ROI + sigmoid kernel ready — same ``state.probe.kernel`` used in disk detection: "
        f"alpha_pr={state.probe_alpha:.6g}, qx0_pr={state.probe_qx0:.6g}, qy0_pr={state.probe_qy0:.6g}",
    )
    return state


class _ProbeKernelShim:
    """Holds kernel samples in ``.data``; use ``probe_kernel_template_ndarray(probe)`` for ``find_Bragg_disks``."""

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=np.float32)


class _ProbeShim:
    """Minimal stand-in for py4DSTEM ``Probe`` used by Step 7 figures and ``find_Bragg_disks``."""

    __slots__ = ("probe", "kernel")

    def __init__(self, probe: np.ndarray, kernel: np.ndarray) -> None:
        self.probe = np.asarray(probe, dtype=np.float32)
        self.kernel = _ProbeKernelShim(kernel)


def format_probe_template_log_line(state: Any) -> str:
    """Human-readable probe template summary for find_Bragg_disks logs."""
    src = getattr(state, "probe_source", None)
    labels = {
        "vacuum_mib": "vacuum 4D scan (shared or Step 2)",
        "bf_roi": "vacuum ROI on main scan (Step 2)",
        "synthetic": "synthetic disk (Step 2)",
        "mean_dp_patch": "mean-DP patch (Step 2)",
        "shared": "shared batch probe",
        "main_workflow": "main workflow probe",
    }
    origin = labels.get(str(src) if src is not None else "", "Step 2 probe (source not recorded)")
    try:
        template = probe_kernel_template_ndarray(getattr(state, "probe", None))
        kshape = getattr(template, "shape", None)
    except Exception as exc:
        kshape = f"invalid ({exc})"
    vac = getattr(state, "vacuum_mib_path", None)
    vac_bit = f" | vacuum_file={vac.name}" if vac is not None else ""
    return (
        f"find_Bragg_disks template=probe kernel 2D ndarray from {origin}; "
        f"probe_source={src!r} | kernel_shape={kshape}{vac_bit}"
    )


def probe_kernel_template_ndarray(probe: Any) -> np.ndarray | None:
    """
    Return a 2D ``ndarray`` for ``find_Bragg_disks(..., template=...)``.

    Passing ``probe.kernel`` when it is a thin Python wrapper (e.g. ``_ProbeKernelShim``) makes
    NumPy treat ``template`` as a 0-D object array; ``np.fft.fft2`` then raises
    ``IndexError: cannot do a non-empty take from an empty axes``.
    """
    if probe is None:
        return None
    kernel = getattr(probe, "kernel", None)
    if kernel is None:
        return None
    if isinstance(kernel, np.ndarray):
        arr: Any = kernel
    else:
        arr = getattr(kernel, "data", kernel)
        for _ in range(6):
            if isinstance(arr, np.ndarray):
                break
            nxt = getattr(arr, "data", None)
            if nxt is None:
                break
            arr = nxt
    out = np.asarray(arr, dtype=np.float64)
    out = np.squeeze(out)
    if out.ndim != 2 or out.size == 0:
        raise RuntimeError(
            "Probe kernel must be a non-empty 2D array for disk detection "
            f"(got shape {getattr(out, 'shape', None)!r})."
        )
    return out


def default_synthetic_disk_radius_px(qshape: Any) -> float:
    try:
        qy, qx = int(qshape[0]), int(qshape[1])
    except Exception:
        return 12.0
    m = max(4, min(qy, qx))
    return float(max(5.0, min(m / 16.0, m / 5.0)))


def _gaussian_disk_qspace(shape_yx: tuple[int, int], center_yx: tuple[float, float], sigma_px: float) -> np.ndarray:
    qy, qx = int(shape_yx[0]), int(shape_yx[1])
    cy, cx = float(center_yx[0]), float(center_yx[1])
    yy, xx = np.indices((qy, qx), dtype=np.float64)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    sig = max(float(sigma_px), 0.5)
    return np.exp(-0.5 * (r / sig) ** 2).astype(np.float32)


def compute_synthetic_disk_probe_step(
    state: WorkflowState,
    radius_px: float | None = None,
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """
    Build a correlation template without a vacuum scan: axisymmetric Gaussian in Q-space plus a
    mild difference-of-Gaussians kernel (edge emphasis). Use when Step 2 vacuum / BF ROI probe
    does not match the BF disk enough for ``find_Bragg_disks`` to mark the central spot.

    ``radius_px`` is roughly the BF disk half-width in detector pixels (sigma is derived from it).
    """
    if state.datacube is None:
        raise RuntimeError("Load datacube before building a synthetic probe.")
    qshape = tuple(int(x) for x in state.datacube.Qshape)
    if len(qshape) < 2:
        raise RuntimeError("datacube.Qshape is invalid.")
    qy, qx = qshape[0], qshape[1]
    cy = 0.5 * float(qy - 1)
    cx = 0.5 * float(qx - 1)
    r_user = float(radius_px) if radius_px is not None else default_synthetic_disk_radius_px(qshape)
    if not np.isfinite(r_user) or r_user <= 0:
        raise ValueError("radius_px must be a positive finite number.")
    sigma = max(0.5, r_user / 2.355)

    probe_img = _gaussian_disk_qspace((qy, qx), (cy, cx), sigma)
    wide = _gaussian_disk_qspace((qy, qx), (cy, cx), sigma * 2.4)
    kernel_img = probe_img.astype(np.float32) - 0.32 * wide.astype(np.float32)
    kernel_img = kernel_img - float(kernel_img.min())
    mx = float(kernel_img.max())
    if mx > 0:
        kernel_img = (kernel_img / mx).astype(np.float32)

    state.reset_probe_products()
    state.probe = _ProbeShim(probe_img, kernel_img)
    state.probe_alpha = float(r_user)
    state.probe_qx0 = float(cy)
    state.probe_qy0 = float(cx)
    state.probe_source = "synthetic"
    state.vacuum_mib_path = None
    _log(
        log,
        "Synthetic axisymmetric probe + kernel (not from vacuum). "
        f"radius_px≈{r_user:.3g} | Qshape={qshape} | centre (row,col)=({cy:.2f},{cx:.2f}). "
        "Tune radius if Step 7 still misses the BF disk.",
    )
    return state


def compute_probe_from_mean_dp_patch_step(
    state: WorkflowState,
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """
    One full ``get_dp_mean`` pass, then ``get_probe_size`` on the mean DP to locate the BF blob,
    crop a square patch, and build a lightweight kernel (unsharp: patch minus Gaussian blur).

    Slower than a synthetic disk but uses the actual mean diffraction pattern shape.
    """
    if state.datacube is None:
        raise RuntimeError("Load datacube before computing a mean-DP patch probe.")

    dc = state.datacube

    def _workflow():
        _log(log, "Mean-DP patch probe: running get_dp_mean (full 4D pass, can take a long time)…")
        dp_mean = dc.get_dp_mean()
        arr = np.asarray(getattr(dp_mean, "data", dp_mean), dtype=np.float64)
        if arr.ndim != 2:
            raise RuntimeError(f"Expected 2D mean DP, got shape {getattr(arr, 'shape', None)}")
        alpha_pr, qrow, qcol = dc.get_probe_size(arr)
        alpha_pr = float(alpha_pr)
        half = int(np.clip(alpha_pr * 3.5, 16.0, min(arr.shape) // 2 - 2))
        ci = int(np.round(float(qrow)))
        cj = int(np.round(float(qcol)))
        i0 = max(0, ci - half)
        i1 = min(arr.shape[0], ci + half + 1)
        j0 = max(0, cj - half)
        j1 = min(arr.shape[1], cj + half + 1)
        if i1 - i0 < 5 or j1 - j0 < 5:
            raise RuntimeError("Mean-DP patch would be too small; check Qshape / data.")
        patch = np.asarray(arr[i0:i1, j0:j1], dtype=np.float32)
        try:
            from scipy.ndimage import gaussian_filter

            blur_sig = max(1.0, float(alpha_pr) * 0.45)
            blur = gaussian_filter(patch, sigma=blur_sig).astype(np.float32)
            kimg = patch - 0.9 * blur
        except Exception:
            kimg = patch.copy()
        kimg = kimg.astype(np.float32)
        kimg = kimg - float(kimg.min())
        mx = float(kimg.max())
        if mx > 0:
            kimg = (kimg / mx).astype(np.float32)
        return patch, kimg, alpha_pr, float(qrow), float(qcol)

    patch, kimg, alpha_pr, qrow, qcol = _with_plot_suppressed(_workflow)

    state.reset_probe_products()
    state.probe = _ProbeShim(patch, kimg)
    state.probe_alpha = float(alpha_pr)
    state.probe_qx0 = float(qrow)
    state.probe_qy0 = float(qcol)
    state.probe_source = "mean_dp_patch"
    state.vacuum_mib_path = None
    _log(
        log,
        "Probe from mean DP patch + unsharp-style kernel ready for disk detection: "
        f"alpha_pr={state.probe_alpha:.6g}, qx0_pr={state.probe_qx0:.6g}, qy0_pr={state.probe_qy0:.6g} | "
        f"patch_shape={patch.shape}",
    )
    return state


def load_virtual_images_step(
    state: WorkflowState,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Step 3 input: load precomputed virtual images from .h5 only."""

    if state.visualcube is None:
        if state.precomputed_h5_path is not None and state.precomputed_h5_path.exists():
            _log(log, "visualcube is empty; retrying .h5 detection before Step 3.")
            state.visualcube = _load_visualcube_from_h5(state.precomputed_h5_path, log=log)
        else:
            raise RuntimeError(
                "No valid visualcube is available. Load an .h5 file containing datacube_root first. "
                "This app will not recompute virtual images from raw data."
            )

    images: dict[str, Any] = {}
    missing: list[str] = []
    for key in VIRTUAL_IMAGE_KEYS:
        try:
            images[key] = _read_tree(state.visualcube, key)
        except Exception as exc:
            missing.append(f"{key}: {exc}")

    if missing:
        raise RuntimeError(
            "The .h5 datacube_root does not contain all required precomputed images. "
            "Missing entries: " + " | ".join(missing)
        )

    state.virtual_images = images
    _log(log, "Loaded precomputed ADF, BF, DP mean, and DP max from datacube_root.")
    return images


def try_load_adf_from_sidecar_h5(
    raw_path: str | Path,
    *,
    h5_path: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> np.ndarray | None:
    """
    Load only the ADF virtual image from a sidecar ``.h5`` (no full datacube).
    Used by the batch Bragg UI to build a gallery without loading every scan into RAM.
    """
    raw = Path(raw_path).expanduser()
    h5 = Path(h5_path).expanduser() if h5_path else _default_sidecar_h5_path(raw)
    if not h5.is_file():
        return None
    try:
        visualcube = _load_visualcube_from_h5(h5, log=log)
        obj = _read_tree(visualcube, "annular_dark_field")
        arr = np.asarray(obj.data if hasattr(obj, "data") else obj)
        if arr.ndim != 2:
            return None
        return arr
    except Exception as exc:
        _log(log, f"ADF preview unavailable for {raw.name}: {exc}")
        return None


def image_array_from_tree(state: WorkflowState, key: str = "annular_dark_field") -> np.ndarray:
    """Return a 2D image array for GUI interactions."""

    if not state.virtual_images:
        load_virtual_images_step(state)
    obj = state.virtual_images[key]
    arr = np.asarray(obj.data if hasattr(obj, "data") else obj)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D image for {key}, got shape {arr.shape}")
    return arr


def set_image_pixel_calibration(
    state: WorkflowState,
    pixel_size: float | None,
    units: str = "px",
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Store optional image pixel calibration for downstream bookkeeping."""

    if pixel_size is None:
        state.image_pixel_size = None
        state.image_pixel_units = units or "px"
        _log(log, "Image pixel calibration skipped.")
        return state
    value = float(pixel_size)
    if value <= 0:
        raise ValueError("Image pixel size must be positive.")
    state.image_pixel_size = value
    state.image_pixel_units = units or "px"
    _log(log, f"Image pixel calibration set: {value:g} {state.image_pixel_units}/px")
    return state


def propagate_r_pixel_calibration(
    state: WorkflowState,
    r_px: float,
    r_units: str = "nm",
    log: Callable[[str], None] | None = None,
) -> None:
    """
    Propagate real-space pixel calibration to every calibration-aware object in state:
      • state.image_pixel_size / image_pixel_units  (for custom matplotlib figures)
      • state.datacube.calibration                  (py4DSTEM internals)
      • state.braggpeaks.calibration                (StrainMap axis labels)
      • each entry in state.virtual_images           (py4DSTEM.show axis labels)

    Call this after entering an R pixel size — e.g. from the Load Dialog or
    from the real-space calibration step — so that ALL subsequent figures use nm axes.

    Mirrors the original notebook pattern::

        datacube.calibration.set_R_pixel_size(R_pixel)
        datacube.calibration.set_R_pixel_units(R_unit)
        im_adf.calibration.set_R_pixel_size(R_pixel)
        im_adf.calibration.set_R_pixel_units(R_unit)
        im_bf.calibration.set_R_pixel_size(R_pixel)
        im_bf.calibration.set_R_pixel_units(R_unit)
    """
    if r_px <= 0:
        return

    # 1. State convenience fields
    state.image_pixel_size = r_px
    state.image_pixel_units = r_units

    def _set_cal(obj):
        try:
            cal = getattr(obj, "calibration", None)
            if cal is not None and hasattr(cal, "set_R_pixel_size"):
                cal.set_R_pixel_size(r_px)
                cal.set_R_pixel_units(r_units)
        except Exception:
            pass

    # 2. Datacube
    _set_cal(getattr(state, "datacube", None))

    # 3. Braggpeaks (StrainMap inherits from this)
    _set_cal(getattr(state, "braggpeaks", None))

    # 4. Virtual images
    for img_obj in (getattr(state, "virtual_images", None) or {}).values():
        _set_cal(img_obj)

    _log(log, f"R pixel calibration propagated: {r_px:g} {r_units}/px → datacube, braggpeaks, virtual images.")


def set_roi_from_bounds(
    state: WorkflowState,
    bounds: tuple[int, int, int, int],
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Store ROI bounds and build a real-space boolean mask."""

    if state.datacube is None:
        raise RuntimeError("Load datacube before setting ROI.")
    x0, x1, y0, y1 = map(int, bounds)
    height, width = map(int, state.datacube.Rshape)
    x0 = max(0, min(x0, width - 1))
    x1 = max(1, min(x1, width))
    y0 = max(0, min(y0, height - 1))
    y1 = max(1, min(y1, height))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Invalid ROI bounds; require x0 < x1 and y0 < y1.")
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    state.roi_bounds = (x0, x1, y0, y1)
    state.roi_mask = mask
    _log(log, f"ROI set: x[{x0}:{x1}] y[{y0}:{y1}] pixels={int(mask.sum())}")
    return state


def set_strain_scan_roi_from_bounds(
    state: WorkflowState,
    bounds: tuple[int, int, int, int],
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Define a real-space sector used only for strain: mask Bragg data outside via mask_in_R(mask=~ROI)."""

    if state.datacube is None:
        raise RuntimeError("Load datacube before setting strain scan ROI.")
    x0, x1, y0, y1 = map(int, bounds)
    height, width = map(int, state.datacube.Rshape)
    x0 = max(0, min(x0, width - 1))
    x1 = max(1, min(x1, width))
    y0 = max(0, min(y0, height - 1))
    y1 = max(1, min(y1, height))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Invalid strain scan ROI bounds; require x0 < x1 and y0 < y1.")
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    state.strain_scan_roi_bounds = (x0, x1, y0, y1)
    state.strain_scan_roi_mask = mask
    _log(log, f"Strain scan sector: x[{x0}:{x1}] y[{y0}:{y1}] pixels={int(mask.sum())}")
    return state


def set_bragg_points(
    state: WorkflowState,
    points: list[tuple[float, float]],
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Store the six notebook-style points used for Bragg detection preview."""

    if len(points) != 6:
        raise ValueError(f"Expected exactly 6 points, got {len(points)}.")
    state.bragg_points = [(float(x), float(y)) for x, y in points]
    state.bragg_rxs = tuple(int(round(x)) for x, _y in state.bragg_points)
    state.bragg_rys = tuple(int(round(y)) for _x, y in state.bragg_points)
    _log(log, f"Bragg points set: rxs={state.bragg_rxs}, rys={state.bragg_rys}")
    return state


def _auto_bragg_path(state: WorkflowState) -> Path:
    if state.braggpeaks_path is not None:
        return state.braggpeaks_path
    if state.raw_mib_path is None:
        raise RuntimeError("Cannot infer braggpeaks output path before loading raw data.")
    return state.raw_mib_path.with_name(f"{state.raw_mib_path.stem}braggpeaks.h5")


def detect_selected_bragg_disks_step(
    state: WorkflowState,
    log: Callable[[str], None] | None = None,
):
    """Run notebook-equivalent detection on the six selected scan positions."""

    if state.datacube is None:
        raise RuntimeError("Load datacube before detecting Bragg disks.")
    if state.probe is None:
        raise RuntimeError("Compute probe before detecting Bragg disks.")
    if len(state.bragg_rxs) != 6 or len(state.bragg_rys) != 6:
        raise RuntimeError("Select exactly 6 Bragg points before preview detection.")

    kwargs = dict(state.detect_params)
    raw_sp = kwargs.get("subpixel", "poly")
    kwargs["subpixel"] = normalize_subpixel_keyword(raw_sp)
    if str(raw_sp).strip().lower() != kwargs["subpixel"]:
        _log(
            log,
            f"subpixel: normalized {raw_sp!r} → {kwargs['subpixel']!r} for py4DSTEM.",
        )
    use_cuda = bool(kwargs.get("CUDA")) or bool(kwargs.get("CUDA_batched"))
    if use_cuda:
        _log(log, f"GPU: preview find_Bragg_disks uses CUDA. ({cupy_device_summary()})")
    _ensure_cupy_current_device_for_thread(kwargs, log=log)
    _log(log, format_probe_template_log_line(state))
    template = probe_kernel_template_ndarray(state.probe)
    # NOTE (confirmed intentional, do not "fix"): data=(rys, rxs) here, not (rxs, rys).
    # An automated review once flagged this order as swapped vs. py4DSTEM's usual
    # (rx, ry) convention — verified against real 6-point picks and confirmed correct
    # for how bragg_rxs/bragg_rys are populated in this codebase (set_bragg_points,
    # above). Leave as-is; this is not the same code path as the axis order used
    # elsewhere for full-scan/streamed detection.
    state.selected_disks = state.datacube.find_Bragg_disks(
        data=(state.bragg_rys, state.bragg_rxs),
        template=template,
        **kwargs,
    )
    _log(log, "Selected-point Bragg disk detection complete.")
    return state.selected_disks


def _close_datacube_memmap(state: WorkflowState) -> None:
    """Close a memmap-backed datacube's file handle before its reference is
    dropped, otherwise the OS keeps the mapping resident. Mirrors
    ``engine._close_memmap_handle`` but is duplicated here (a few lines) so
    ``pipeline`` stays free of an ``import engine`` cycle (engine imports
    pipeline lazily, never the reverse)."""
    dc = getattr(state, "datacube", None)
    try:
        data = getattr(dc, "data", None)
        base = getattr(data, "base", None)
        for obj in (data, base):
            mm = getattr(obj, "_mmap", None) or (
                obj if "mmap" in type(obj).__name__.lower() else None
            )
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
    except Exception:
        pass


def _datacube_is_memmap(state: WorkflowState) -> bool:
    """True when the loaded datacube is memmap-backed — a proxy for "large":
    ``_pick_mib_mem_mode`` / MEMMAP import chose memmap precisely because the cube
    did not comfortably fit in RAM."""
    dc = getattr(state, "datacube", None)
    data = getattr(dc, "data", None)
    for obj in (data, getattr(data, "base", None)):
        if obj is not None and "memmap" in type(obj).__name__.lower():
            return True
    return False


def _should_stream_braggpeaks(state: WorkflowState, kwargs: dict) -> bool:
    """Decide whether to run bounded-RAM streaming detection instead of the
    full-scan ``find_Bragg_disks`` -> in-RAM ``BraggVectors`` build.

    Residency-only decision — the detection parameters (and therefore the peaks)
    are identical either way:

    - GPU path (``CUDA`` / ``CUDA_batched``) always uses full-scan: py4DSTEM's
      batched-GPU path is the throughput winner, and the position-list
      ``data=(...)`` streaming call is a CPU / low-RAM lever, not a GPU one.
    - ``FAST4D_STREAM_BRAGG=1`` forces streaming; ``=0`` forces full-scan.
    - Otherwise stream only when the cube is memmap-backed (large enough that the
      loader chose not to hold it in RAM).
    """
    if bool(kwargs.get("CUDA")) or bool(kwargs.get("CUDA_batched")):
        return False
    flag = os.environ.get("FAST4D_STREAM_BRAGG", "").strip()
    if flag == "1":
        return True
    if flag == "0":
        return False
    return _datacube_is_memmap(state)


def compute_braggpeaks_step(
    state: WorkflowState,
    save_path: str | Path | None = None,
    log: Callable[[str], None] | None = None,
):
    """Compute full-scan braggpeaks and save them to HDF5."""

    if state.use_existing_braggpeaks and state.braggpeaks is not None:
        _log(log, "Existing braggpeaks are loaded; skipping full recomputation.")
        return state.braggpeaks
    if state.datacube is None:
        raise RuntimeError("Load datacube before computing braggpeaks.")
    if state.probe is None:
        raise RuntimeError("Compute probe before computing braggpeaks.")

    kwargs = dict(state.detect_params)
    raw_sp = kwargs.get("subpixel", "poly")
    kwargs["subpixel"] = normalize_subpixel_keyword(raw_sp)
    if str(raw_sp).strip().lower() != kwargs["subpixel"]:
        _log(
            log,
            f"subpixel: normalized {raw_sp!r} → {kwargs['subpixel']!r} for py4DSTEM.",
        )
    use_cuda = bool(kwargs.get("CUDA")) or bool(kwargs.get("CUDA_batched"))
    _ensure_cupy_current_device_for_thread(kwargs, log=log)
    template = probe_kernel_template_ndarray(state.probe)
    out_path = Path(save_path).expanduser() if save_path else _auto_bragg_path(state)

    if _should_stream_braggpeaks(state, kwargs):
        # Bounded-RAM path: detect in position-batches and stream peaks to a temp
        # file, release the raw cube, THEN assemble a py4DSTEM-native BraggVectors
        # from disk and save it. Same detection params -> same peaks (parity locked
        # by tests/test_bragg_stream_detect.py); only residency differs. Reuses the
        # library's position-scoped find_Bragg_disks(data=(rxs, rys)) call — no
        # custom detector, no science change.
        from bragg_stream import (
            detect_braggpeaks_streaming,
            finalize_stream_to_braggvectors,
        )

        q_shape = tuple(int(v) for v in state.datacube.Qshape)
        _log(log, f"Computing streamed braggpeaks (bounded RAM) with params: {kwargs}")
        _log(log, format_probe_template_log_line(state))
        stream_tmp = out_path.with_name(out_path.stem + ".stream.h5")
        detect_braggpeaks_streaming(
            state.datacube, template, stream_tmp, log=log, **kwargs
        )
        # Release the raw cube BEFORE materializing the full peak list, so peak RAM
        # never holds the tens-of-GB cube and the full PointListArray at once — this
        # is exactly what streaming buys over the full-scan build below.
        _close_datacube_memmap(state)
        state.datacube = None
        import gc
        gc.collect()
        state.braggpeaks = finalize_stream_to_braggvectors(
            stream_tmp, Qshape=q_shape, log=log
        )
        save_braggpeaks_file(state.braggpeaks, out_path, log=log)
        try:
            stream_tmp.unlink()
        except Exception:
            pass
        state.braggpeaks_path = out_path
        _log(log, f"Full-scan braggpeaks ready (streamed): {out_path}")
        return state.braggpeaks

    # Full-scan path (default; GPU-batched throughput when CUDA is enabled).
    if use_cuda:
        _log(
            log,
            "GPU: find_Bragg_disks CUDA/CUDA_batched enabled — nvidia-smi should rise during this step. "
            f"({cupy_device_summary()})",
        )
    else:
        _log(log, "GPU: find_Bragg_disks CUDA=False — full scan runs on CPU only (nvidia-smi may stay flat).")
    _log(log, f"Computing full-scan braggpeaks with params: {kwargs}")
    _log(log, format_probe_template_log_line(state))
    state.braggpeaks = state.datacube.find_Bragg_disks(
        template=template,
        **kwargs,
    )

    try:
        save_braggpeaks_datacube_notebook_style(state, out_path, log=log)
    except Exception as exc:
        _log(
            log,
            f"Notebook-style save (datacube, tree=None) failed ({type(exc).__name__}: {exc}); "
            "falling back to braggpeaks-only HDF5.",
        )
        save_braggpeaks_file(state.braggpeaks, out_path, log=log)
    state.braggpeaks_path = out_path
    _log(log, f"Full-scan braggpeaks ready: {out_path}")
    # Peaks are now detected AND persisted to ``out_path``. The full raw datacube
    # (tens of GB) and the braggpeaks were both alive at once during the save above
    # (the one designed-in double-residency point); release the cube now that it is
    # no longer needed. Everything downstream — calibration, strain, stress — runs
    # on the compact braggpeaks/BVM layer (Path A) and never touches the raw cube
    # again. If a Path-B step is revisited, ``engine.load_datacube`` re-loads it from
    # ``raw_path``. This is a pure lifecycle change: the detection math already ran
    # and its result (``state.braggpeaks``) + the on-disk .h5 are untouched.
    _close_datacube_memmap(state)
    state.datacube = None
    # A single gc.collect() reclaims the freed cube and any transient copies
    # py4DSTEM's save path made before the next step runs.
    import gc
    gc.collect()
    return state.braggpeaks


def save_braggpeaks_datacube_notebook_style(
    state: WorkflowState,
    path: str | Path,
    log: Callable[[str], None] | None = None,
) -> Path:
    """
    Match the notebook pattern: ``py4DSTEM.save(path, datacube, tree=None, mode='o')``
    so the on-disk tree matches interactive workflows.
    """
    import py4DSTEM

    if state.datacube is None:
        raise RuntimeError("No datacube to save.")
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gone = _unlink_existing_with_retries(out_path)
    if out_path.exists() and not gone:
        _log(
            log,
            "Could not remove existing braggpeaks file after retries "
            f"({out_path}); writing via temp + os.replace instead.",
        )

    dc = state.datacube

    def _save_destination(destination: Path) -> None:
        last_type_error: TypeError | None = None
        for kwargs in (
            {"tree": None, "mode": "o"},
            {"tree": None, "mode": "w"},
            {"tree": None},
            {},
        ):
            try:
                py4DSTEM.save(str(destination), dc, **kwargs)
                return
            except TypeError as exc:
                last_type_error = exc
                continue
        try:
            py4DSTEM.save(str(destination), [dc], mode="w")
            return
        except TypeError:
            pass
        if last_type_error is not None:
            raise last_type_error
        raise RuntimeError("py4DSTEM.save(datacube, ...) is incompatible with this py4DSTEM version.")

    def _unlink_quiet(destination: Path) -> None:
        try:
            if destination.exists():
                destination.unlink(missing_ok=True)
        except TypeError:
            try:
                if destination.exists():
                    destination.unlink()
            except Exception:
                pass
        except Exception:
            pass

    last_err: BaseException | None = None
    if not out_path.exists():
        try:
            _save_destination(out_path)
            _log(log, f"Saved braggpeaks context (datacube, notebook style): {out_path}")
            return out_path
        except BaseException as exc:
            last_err = exc

    tmp_path = out_path.with_name(out_path.name + "." + uuid.uuid4().hex[:12] + ".partial.h5")
    try:
        _save_destination(tmp_path)
        os.replace(tmp_path, out_path)
        _log(log, f"Saved braggpeaks context (datacube, notebook style, temp replace): {out_path}")
        return out_path
    except BaseException as exc:
        last_err = exc
    finally:
        _unlink_quiet(tmp_path)

    if last_err is not None:
        raise RuntimeError("Could not save datacube (notebook style) to HDF5.") from last_err
    raise RuntimeError("Could not save datacube (notebook style) (unknown failure).")


def _unlink_existing_with_retries(target: Path, *, attempts: int = 10, delay_s: float = 0.06) -> bool:
    """Return True when ``target`` is absent afterward (already missing or deleted)."""

    def _unlink_one() -> bool:
        """Return True only if file was removed."""
        try:
            if not target.exists():
                return False
            target.unlink()
            return True
        except Exception:
            return False

    for i in range(attempts):
        if not target.exists():
            return True
        removed = _unlink_one()
        if removed and not target.exists():
            return True
        if i < attempts - 1:
            time.sleep(delay_s)
    return not target.exists()


def _detach_braggpeaks_for_export(braggpeaks: Any):
    """
    Prefer a saver-friendly duplicate when EM tree linkage prevents py4DSTEM/emdfile
    from matching runtime nodes during save. Prefer cut_from_tree; else copy()—avoid deepcopy.

    Returning the original object signals callers they may reuse it for in-memory workflows.
    """
    cut_from_tree = getattr(braggpeaks, "cut_from_tree", None)
    if callable(cut_from_tree):
        try:
            detached = cut_from_tree()
            if detached is not None:
                return detached
        except Exception:
            pass
    cp = getattr(braggpeaks, "copy", None)
    if callable(cp):
        try:
            duplicated = cp()
            if duplicated is not None and duplicated is not braggpeaks:
                return duplicated
        except Exception:
            pass
    return braggpeaks


def save_braggpeaks_file(braggpeaks, path: str | Path, log: Callable[[str], None] | None = None) -> Path:
    import py4DSTEM

    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale outputs so emdfile does not merge into an incompatible on-disk tree.
    gone = _unlink_existing_with_retries(out_path)
    if out_path.exists() and not gone:
        _log(
            log,
            "Could not remove existing braggpeaks file after retries "
            f"({out_path}); writing via temp + os.replace instead.",
        )

    detached = _detach_braggpeaks_for_export(braggpeaks)
    payloads: list[Any] = [braggpeaks]
    if detached is not braggpeaks:
        payloads.append(detached)

    last_err: BaseException | None = None

    def _save(destination: Path, payload: Any) -> None:
        py4DSTEM.save(str(destination), [payload], mode="w")

    def _unlink_quiet(destination: Path) -> None:
        try:
            if destination.exists():
                destination.unlink(missing_ok=True)
        except TypeError:
            try:
                if destination.exists():
                    destination.unlink()
            except Exception:
                pass
        except Exception:
            pass

    # 1) Prefer writing in place once the path is writable and empty.
    if not out_path.exists():
        for pl in payloads:
            try:
                _save(out_path, pl)
                _log(log, f"Saved braggpeaks file: {out_path}")
                return out_path
            except BaseException as exc:
                last_err = exc
        if last_err is not None:
            _log(log, f"Direct save failed ({last_err}); trying temp atomic replace.")

    # 2) Temp file next to destination, then rename (helps Windows / OneDrive lock on unlink or path).
    for pl in payloads:
        tmp_path = out_path.with_name(out_path.name + "." + uuid.uuid4().hex[:12] + ".partial.h5")
        try:
            _save(tmp_path, pl)
            os.replace(tmp_path, out_path)
            _log(log, f"Saved braggpeaks file (temp atomic replace): {out_path}")
            return out_path
        except BaseException as exc:
            last_err = exc
        finally:
            _unlink_quiet(tmp_path)

    if last_err is not None:
        raise RuntimeError("Could not save braggpeaks to HDF5 (see traceback).") from last_err
    raise RuntimeError("Could not save braggpeaks (unknown failure).")


def _collect_bragg_candidates(obj: Any, depth: int = 0, seen: set[int] | None = None) -> list[Any]:
    if seen is None:
        seen = set()
    if obj is None or depth > 5:
        return []
    oid = id(obj)
    if oid in seen:
        return []
    seen.add(oid)

    found: list[Any] = []
    if hasattr(obj, "measure_origin") and hasattr(obj, "fit_origin"):
        found.append(obj)
    if isinstance(obj, dict):
        for value in obj.values():
            found.extend(_collect_bragg_candidates(value, depth + 1, seen))
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found.extend(_collect_bragg_candidates(value, depth + 1, seen))
    if hasattr(obj, "tree") and callable(getattr(obj, "tree")):
        for key in ("braggpeaks", "braggvectors", "datacube_root"):
            try:
                found.extend(_collect_bragg_candidates(obj.tree(key), depth + 1, seen))
            except Exception:
                pass
    try:
        values = vars(obj).values()
    except Exception:
        values = []
    for value in values:
        found.extend(_collect_bragg_candidates(value, depth + 1, seen))
    return found


def _rshape_tuple(obj: Any) -> tuple[int, ...] | None:
    try:
        return tuple(int(v) for v in obj.Rshape)
    except Exception:
        return None


def _bragg_candidate_score(obj: Any, expected_rshape: tuple[int, ...] | None = None) -> tuple[int, int]:
    rshape = _rshape_tuple(obj)
    exact = int(expected_rshape is not None and rshape == tuple(expected_rshape))
    size = int(np.prod(rshape)) if rshape else 0
    return exact, size


def load_braggpeaks_file(
    path: str | Path,
    expected_rshape=None,
    log: Callable[[str], None] | None = None,
):
    """Load an existing braggpeaks HDF5 using common py4DSTEM layouts."""

    import py4DSTEM

    bragg_path = Path(path).expanduser()
    if not bragg_path.exists():
        raise FileNotFoundError(f"braggpeaks file does not exist: {bragg_path}")

    # Datapaths to try, in order. The literal ``None`` (auto-detect the single
    # EMD root) is first for backward compatibility, but emdfile's auto-detect
    # occasionally raises "dictionary changed size during iteration" on the FIRST
    # read of a file — a nondeterministic dict-iteration bug in Root.from_h5 that
    # a retry clears. We therefore (a) also enumerate the file's actual top-level
    # HDF5 groups (EMD root groups) as explicit datapaths — e.g. a streamed result
    # finalized to a py4DSTEM ``BraggVectors`` lands under a ``braggvectors_root``
    # group whose name none of the hardcoded guesses match — and (b) retry ``None``
    # once at the end so the flaky-first-read case still resolves.
    try:
        import h5py

        with h5py.File(str(bragg_path), "r") as _f:
            root_group_names = [k for k in _f.keys() if isinstance(_f[k], h5py.Group)]
    except Exception:
        root_group_names = []

    datapaths: list[Any] = [None, "datacube_root", "braggpeaks", "braggvectors", "root"]
    for name in root_group_names:
        if name not in datapaths:
            datapaths.append(name)
    datapaths.append(None)  # retry auto-detect last (dodges the flaky first read)

    bragg_candidates: list[Any] = []
    for datapath in datapaths:
        try:
            if datapath is None:
                obj = py4DSTEM.read(filepath=str(bragg_path))
            else:
                obj = py4DSTEM.read(filepath=str(bragg_path), datapath=datapath)
            candidates = _collect_bragg_candidates(obj)
            if candidates:
                bragg_candidates.extend(candidates)
                break   # stop as soon as we found valid braggpeaks
        except Exception:
            pass

    if not bragg_candidates:
        raise RuntimeError(f"No valid braggpeaks object found in: {bragg_path}")

    expected = tuple(int(v) for v in expected_rshape) if expected_rshape is not None else None
    if expected is not None:
        matching = [obj for obj in bragg_candidates if _rshape_tuple(obj) == expected]
        if matching:
            bragg_candidates = matching
        else:
            shapes = sorted({_rshape_tuple(obj) for obj in bragg_candidates})
            raise RuntimeError(
                f"Found Bragg-like objects, but none match datacube Rshape={expected}. "
                f"Candidate Rshapes={shapes}. Select the full-scan braggpeaks file."
            )

    found = max(bragg_candidates, key=lambda obj: _bragg_candidate_score(obj, expected))
    try:
        found.setcal()
    except Exception:
        pass
    _log(log, f"Loaded full-scan braggpeaks from: {bragg_path}")
    _log(log, f"braggpeaks Rshape: {_rshape_tuple(found)} | calstate={getattr(found, 'calstate', None)}")
    return found


def require_braggpeaks(state: WorkflowState):
    if state.braggpeaks is None:
        raise RuntimeError("Load or compute braggpeaks before origin correction.")
    if not (hasattr(state.braggpeaks, "measure_origin") and hasattr(state.braggpeaks, "fit_origin")):
        raise RuntimeError(f"Current braggpeaks object is not valid for origin correction: {type(state.braggpeaks).__name__}")
    if state.datacube is not None:
        expected = tuple(int(v) for v in state.datacube.Rshape)
        actual = _rshape_tuple(state.braggpeaks)
        if actual != expected:
            raise RuntimeError(
                f"Loaded braggpeaks Rshape={actual} does not match datacube Rshape={expected}. "
                "Origin correction requires the full-scan braggpeaks object."
            )
    return state.braggpeaks


def _extract_xy_from_pointlist(pointlist):
    data = getattr(pointlist, "data", pointlist)
    if hasattr(data, "dtype") and data.dtype.names is not None:
        names = list(data.dtype.names)
        for xname, yname in (("qx", "qy"), ("x", "y"), ("q_x", "q_y"), ("Qx", "Qy"), ("X", "Y")):
            if xname in names and yname in names:
                return np.asarray(data[xname]), np.asarray(data[yname])
        raise RuntimeError(f"Could not find coordinate fields in pointlist: {names}")

    for xname, yname in (("qx", "qy"), ("x", "y")):
        if hasattr(pointlist, xname) and hasattr(pointlist, yname):
            return np.asarray(getattr(pointlist, xname)), np.asarray(getattr(pointlist, yname))
    raise RuntimeError("PointList does not expose recognizable x/y coordinates.")


def make_bvm_raw_from_pointlistarray(
    braggpeaks,
    sampling: int = 1,
    *,
    use_cupy_histogram: bool = False,
    log: Callable[[str], None] | None = None,
) -> np.ndarray:
    if hasattr(braggpeaks, "shape"):
        nx, ny = braggpeaks.shape[:2]
    elif hasattr(braggpeaks, "size_R"):
        nx, ny = braggpeaks.size_R
    elif hasattr(braggpeaks, "Rshape"):
        nx, ny = braggpeaks.Rshape
    else:
        raise RuntimeError("Could not infer scan shape from the loaded object.")

    xs_all = []
    ys_all = []
    for i in range(int(nx)):
        for j in range(int(ny)):
            pointlist = braggpeaks.get_pointlist(i, j)
            x, y = _extract_xy_from_pointlist(pointlist)
            xs_all.append(np.asarray(x) * int(sampling))
            ys_all.append(np.asarray(y) * int(sampling))

    if not xs_all:
        return np.zeros((1, 1))

    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)
    xmin = int(np.floor(xs_all.min())) - 2
    xmax = int(np.ceil(xs_all.max())) + 2
    ymin = int(np.floor(ys_all.min())) - 2
    ymax = int(np.ceil(ys_all.max())) + 2
    xbins = np.arange(xmin, xmax + 2)
    ybins = np.arange(ymin, ymax + 2)
    if bool(use_cupy_histogram):
        try:
            import cupy as cp  # type: ignore

            xs = cp.asarray(xs_all, dtype=cp.float32)
            ys = cp.asarray(ys_all, dtype=cp.float32)
            xb = cp.asarray(xbins, dtype=cp.float32)
            yb = cp.asarray(ybins, dtype=cp.float32)
            hist, _yedges, _xedges = cp.histogram2d(ys, xs, bins=[yb, xb])
            _log(log, f"make_bvm_raw_from_pointlistarray: GPU histogram2d via CuPy (n={int(xs_all.size)})")
            return cp.asnumpy(hist)
        except Exception as exc:
            _log(log, f"make_bvm_raw_from_pointlistarray: CuPy histogram fallback to NumPy ({exc})")

    hist, _yedges, _xedges = np.histogram2d(ys_all, xs_all, bins=[ybins, xbins])
    return hist


def get_bvm_raw(
    braggpeaks,
    sampling: int = 1,
    *,
    use_cupy_histogram: bool = False,
    log: Callable[[str], None] | None = None,
):
    if hasattr(braggpeaks, "histogram"):
        _log(log, "get_bvm_raw: using py4DSTEM braggpeaks.histogram(mode='raw') (device depends on py4DSTEM build).")
        return braggpeaks.histogram(mode="raw", sampling=int(sampling))
    return make_bvm_raw_from_pointlistarray(
        braggpeaks,
        sampling=int(sampling),
        use_cupy_histogram=bool(use_cupy_histogram),
        log=log,
    )


def compute_bvm_raw_step(
    state: WorkflowState,
    sampling: int = 1,
    log: Callable[[str], None] | None = None,
):
    """Build raw BVM histogram for picking center_guess."""

    if state.braggpeaks is None:
        raise RuntimeError("Load or compute braggpeaks before origin correction.")
    braggpeaks = state.braggpeaks
    sampling = int(max(1, sampling))
    state.origin_sampling = sampling
    _log(log, f"Bragg object type: {type(braggpeaks)}")
    _log(log, f"Bragg object attrs: {summarize_obj(braggpeaks)}")
    _log(log, f"Bragg object Rshape: {_rshape_tuple(braggpeaks)} | calstate={getattr(braggpeaks, 'calstate', None)}")
    use_cupy = bool((state.bvm_params or {}).get("use_cupy_histogram", False))
    state.bvm_raw = get_bvm_raw(braggpeaks, sampling=sampling, use_cupy_histogram=use_cupy, log=log)
    _log(log, f"Computed raw BVM for origin pick with sampling={sampling} using notebook-equivalent get_bvm_raw().")
    return state.bvm_raw


def center_guess_qxy(center_guess_yx: tuple[int, int]) -> tuple[int, int]:
    """GUI stores ``(y, x)`` from the BVM picker; py4DSTEM ``measure_origin`` expects ``(qx, qy)``.

    The GUI's ``y`` is the diffraction row (== ``qx``) and ``x`` is the column
    (== ``qy``), so the stored ``(y, x)`` already matches py4DSTEM's ``(qx, qy)``
    ordering — it must be passed straight through. (The previous ``return (x, y)``
    swap inverted the handoff, so a pick of ``(151, 131)`` was calibrated as
    ``(131, 151)``.)
    """
    y, x = map(int, center_guess_yx)
    return (y, x)


def set_origin_center_guess(
    state: WorkflowState,
    center_guess_yx: tuple[int, int],
    sampling: int | None = None,
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Store center_guess as native ``(y, x)`` from the GUI (BVM pick / manual fields)."""

    y, x = map(int, center_guess_yx)
    state.center_guess = (y, x)
    if sampling is not None:
        state.origin_sampling = int(max(1, sampling))
    _log(log, f"center_guess set to native (y,x)={state.center_guess}, sampling={state.origin_sampling}")
    return state


def run_origin_correction_step(
    state: WorkflowState,
    sampling: int | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run measure_origin, fit_origin, and setcal on the current braggpeaks."""

    braggpeaks = require_braggpeaks(state)
    if state.center_guess is None:
        raise RuntimeError("Pick center_guess before running origin correction.")

    if sampling is not None:
        state.origin_sampling = int(max(1, sampling))
    sampling = int(max(1, state.origin_sampling))

    state.bvm_raw = braggpeaks.histogram(mode="raw", sampling=sampling)

    cg_qxy = center_guess_qxy(state.center_guess)
    _log(
        log,
        f"Running measure_origin(center_guess GUI y,x={state.center_guess} → py4DSTEM qx,qy={cg_qxy})...",
    )
    origin_measurement = braggpeaks.measure_origin(center_guess=cg_qxy)
    _log(log, "Running fit_origin()...")
    qx0_fit, qy0_fit, qx0_residuals, qy0_residuals = braggpeaks.fit_origin()
    braggpeaks.setcal()
    state.bvm_centered = braggpeaks.histogram(mode="cal", sampling=sampling)

    state.origin_measurement = origin_measurement
    state.origin_fit = (qx0_fit, qy0_fit, qx0_residuals, qy0_residuals)
    _log(log, f"Origin correction complete. calstate={getattr(braggpeaks, 'calstate', None)}")
    return {
        "origin_measurement": origin_measurement,
        "qx0_fit": qx0_fit,
        "qy0_fit": qy0_fit,
        "qx0_residuals": qx0_residuals,
        "qy0_residuals": qy0_residuals,
        "bvm_raw": state.bvm_raw,
        "bvm_centered": state.bvm_centered,
    }


ORIGIN_BVM_VIS_PARAMS = {
    "scaling": "power",
    "power": 0.5,
    "intensity_range": "absolute",
    "vmin": 0,
    "vmax": 2e3,
}


# Match py4DSTEM notebook origin-fit panels (pixels in Q-space).
ORIGIN_RESIDUAL_PLOT_RANGE = 2.0


def _origin_residual_display_limits(
    arr: np.ndarray,
    *,
    plot_range: float = ORIGIN_RESIDUAL_PLOT_RANGE,
) -> tuple[float, float] | tuple[None, None]:
    """Robust symmetric auto-scale around zero: ±nanpercentile(|residual|, 99).

    Display-only — does not alter the residual values. Falls back to ±plot_range
    (default ±2 px) when the 99th percentile is 0, NaN or infinite (e.g. all-zero
    or degenerate residuals).
    """
    f = np.asarray(arr, dtype=float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return None, None
    lim = float(np.nanpercentile(np.abs(f), 99))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(max(0.1, plot_range))
    return -lim, lim


def build_origin_correction_result_figure(
    state: WorkflowState,
    result: dict[str, Any],
    *,
    log: Callable[[str], None] | None = None,
):
    """2×3 matplotlib figure: BVM panels + qx/qy residuals (same layout as single-scan Step 9)."""
    import py4DSTEM
    from matplotlib.figure import Figure

    bvm_raw = result["bvm_raw"]
    bvm_centered = result["bvm_centered"]
    qx0_residuals = np.asarray(result["qx0_residuals"])
    qy0_residuals = np.asarray(result["qy0_residuals"])
    sampling = int(max(1, getattr(state, "origin_sampling", 1) or 1))

    fig = Figure(figsize=(12, 8), constrained_layout=True)
    axes = fig.subplots(2, 3).ravel()

    py4DSTEM.show(
        bvm_raw,
        figax=(fig, axes[0]),
        title=f"BVM raw (sampling={sampling})",
        **ORIGIN_BVM_VIS_PARAMS,
    )

    points_for_plot_raw = None
    if state.center_guess is not None:
        y_native, x_native = state.center_guess
        qx_bvm = int(round(y_native * sampling))
        qy_bvm = int(round(x_native * sampling))
        points_for_plot_raw = {"x": qx_bvm, "y": qy_bvm}
    show_kwargs = dict(ORIGIN_BVM_VIS_PARAMS)
    if points_for_plot_raw is not None:
        show_kwargs["points"] = points_for_plot_raw
    py4DSTEM.show(
        bvm_raw,
        figax=(fig, axes[1]),
        title="BVM raw + center_guess",
        **show_kwargs,
    )

    py4DSTEM.show(
        bvm_centered,
        figax=(fig, axes[2]),
        title="BVM centered",
        **ORIGIN_BVM_VIS_PARAMS,
    )

    py4DSTEM.show(
        bvm_centered,
        figax=(fig, axes[3]),
        title="BVM centered zoom near origin",
        vmin=0,
        vmax=1,
        circle={"center": bvm_centered.origin, "R": 2},
    )
    try:
        oy, ox = bvm_centered.origin
        zoom_r = 1.5 * int(max(1, sampling))
        axes[3].set_xlim(ox - zoom_r, ox + zoom_r)
        axes[3].set_ylim(oy + zoom_r, oy - zoom_r)
    except Exception:
        pass

    qx_f = qx0_residuals[np.isfinite(qx0_residuals)]
    qy_f = qy0_residuals[np.isfinite(qy0_residuals)]
    qx_vmin, qx_vmax = _origin_residual_display_limits(qx0_residuals)
    qy_vmin, qy_vmax = _origin_residual_display_limits(qy0_residuals)

    rh, rw = int(qx0_residuals.shape[0]), int(qx0_residuals.shape[1])
    try:
        qh, qw = int(np.asarray(bvm_raw.data if hasattr(bvm_raw, "data") else bvm_raw).shape[0]), int(
            np.asarray(bvm_raw.data if hasattr(bvm_raw, "data") else bvm_raw).shape[1]
        )
    except Exception:
        qh, qw = 0, 0
    r_note = f"R-space {rh}×{rw}"
    q_note = f"Q-space BVM ~{qh}×{qw} (sampling={sampling})" if qh and qw else f"sampling={sampling}"

    im4 = axes[4].imshow(
        qx0_residuals,
        cmap="coolwarm",
        vmin=qx_vmin,
        vmax=qx_vmax,
        aspect="equal",
        origin="upper",
    )
    qx_lim = qx_vmax if qx_vmax is not None else ORIGIN_RESIDUAL_PLOT_RANGE
    axes[4].set_title(f"qx residuals ({r_note}, ±{qx_lim:.2g} px)")
    fig.colorbar(im4, ax=axes[4], fraction=0.046)

    im5 = axes[5].imshow(
        qy0_residuals,
        cmap="coolwarm",
        vmin=qy_vmin,
        vmax=qy_vmax,
        aspect="equal",
        origin="upper",
    )
    qy_lim = qy_vmax if qy_vmax is not None else ORIGIN_RESIDUAL_PLOT_RANGE
    axes[5].set_title(f"qy residuals ({r_note}, ±{qy_lim:.2g} px)")
    fig.colorbar(im5, ax=axes[5], fraction=0.046)

    _log(log, f"Origin figure: {q_note}; residuals are per scan pixel in {r_note} (not Q-space).")
    if qx_f.size:
        _log(
            log,
            "qx residuals: "
            f"mean={float(qx_f.mean()):.4g}, std={float(qx_f.std()):.4g}, "
            f"min={float(qx_f.min()):.4g}, max={float(qx_f.max()):.4g}, "
            f"color scale ±{ORIGIN_RESIDUAL_PLOT_RANGE:g} px (py4DSTEM default)",
        )
    if qy_f.size:
        _log(
            log,
            "qy residuals: "
            f"mean={float(qy_f.mean()):.4g}, std={float(qy_f.std()):.4g}, "
            f"min={float(qy_f.min()):.4g}, max={float(qy_f.max()):.4g}, "
            f"color scale ±{ORIGIN_RESIDUAL_PLOT_RANGE:g} px (py4DSTEM default)",
        )
    mx = max(
        float(np.nanmax(np.abs(qx_f))) if qx_f.size else 0.0,
        float(np.nanmax(np.abs(qy_f))) if qy_f.size else 0.0,
    )
    if mx > ORIGIN_RESIDUAL_PLOT_RANGE * 1.5:
        _log(
            log,
            f"Note: |residual| up to {mx:.4g} px — check center_guess or Bragg disk quality "
            f"(display still ±{ORIGIN_RESIDUAL_PLOT_RANGE:g} like the notebook).",
        )

    return fig


def _braggpeaks_for_roi_from_bp(braggpeaks, state: WorkflowState, use_roi: bool = True):
    if use_roi and state.roi_mask is not None:
        try:
            return braggpeaks.mask_in_R(mask=~state.roi_mask)
        except Exception:
            return braggpeaks
    return braggpeaks


def _braggpeaks_for_roi(state: WorkflowState, use_roi: bool = True):
    return _braggpeaks_for_roi_from_bp(require_braggpeaks(state), state, use_roi=use_roi)


def _mask_braggpeaks_keep_scan_sector(braggpeaks, roi_inside: np.ndarray, log: Callable[[str], None] | None = None):
    """Keep diffraction data only at scan positions where roi_inside is True (notebook: mask_in_R(mask=~ROI))."""
    m = np.asarray(roi_inside, dtype=bool)
    rshape = getattr(braggpeaks, "Rshape", None)
    if rshape is None:
        rshape = getattr(braggpeaks, "rshape", None)
    if rshape is None:
        raise RuntimeError("braggpeaks has no Rshape; cannot mask_in_R.")
    rh, rw = int(rshape[0]), int(rshape[1])
    if m.shape != (rh, rw):
        raise RuntimeError(
            f"strain_scan_roi_mask shape {m.shape} does not match braggpeaks Rshape {(rh, rw)}."
        )
    if not np.any(m):
        raise RuntimeError("Strain scan ROI mask is empty.")
    n_in = int(m.sum())
    _log(log, f"strain: mask_in_R(mask=~sector) keeps {n_in}/{m.size} scan positions inside sector")
    try:
        out = braggpeaks.mask_in_R(mask=~m)
    except Exception as exc:
        raise RuntimeError(f"mask_in_R failed: {exc}") from exc
    if out is None:
        raise RuntimeError("mask_in_R returned None.")
    return out


def compute_ellipse_bvm_step(
    state: WorkflowState,
    sampling: int = 1,
    use_roi: bool = True,
    log: Callable[[str], None] | None = None,
):
    """Compute calibrated BVM used by the optional ellipse fit."""

    sampling = int(max(1, sampling))
    state.ellipse_sampling = sampling
    state.ellipse_use_roi = bool(use_roi)
    braggpeaks_use = _braggpeaks_for_roi(state, use_roi=state.ellipse_use_roi)
    state.ellipse_bvm = braggpeaks_use.histogram(mode="cal", sampling=sampling)
    _log(log, f"Computed ellipse BVM mode='cal', sampling={sampling}, use_roi={state.ellipse_use_roi}.")
    return state.ellipse_bvm


def fit_ellipse_step(
    state: WorkflowState,
    q_range: tuple[int, int],
    sampling: int = 1,
    use_roi: bool = True,
    center: tuple[float, float] | None = None,
    log: Callable[[str], None] | None = None,
):
    """Fit ellipse on calibrated BVM using py4DSTEM fit_ellipse_1D.

    ``center``, if given, is a ``(y, x)`` BVM-pixel point used as the
    annulus/initial-guess center for the fit instead of ``bvm.origin``.
    """

    import py4DSTEM

    r0, r1 = sorted(tuple(map(int, q_range)))
    if r0 < 0 or r1 <= r0:
        raise ValueError("Ellipse q_range must satisfy 0 <= r0 < r1.")
    state.ellipse_q_range = (r0, r1)
    bvm = compute_ellipse_bvm_step(state, sampling=sampling, use_roi=use_roi, log=log)
    fit_center = tuple(map(float, center)) if center is not None else bvm.origin
    state.p_ellipse = py4DSTEM.process.calibration.fit_ellipse_1D(
        bvm,
        center=fit_center,
        fitradii=(r0, r1),
    )
    if state.p_ellipse is None:
        raise RuntimeError("fit_ellipse_1D returned None.")
    _log(log, f"Ellipse fit complete. p_ellipse={state.p_ellipse}")
    return {"bvm": bvm, "p_ellipse": state.p_ellipse, "q_range": state.ellipse_q_range}


def apply_ellipse_step(state: WorkflowState, log: Callable[[str], None] | None = None):
    """Apply fitted ellipse calibration in-place to braggpeaks."""

    braggpeaks = require_braggpeaks(state)
    if state.p_ellipse is None:
        raise RuntimeError("Fit ellipse before applying it.")
    braggpeaks.calibration.set_p_ellipse(state.p_ellipse)
    braggpeaks.setcal()
    _log(log, f"Applied ellipse calibration. calstate={getattr(braggpeaks, 'calstate', None)}")
    return braggpeaks


def set_q_pixel_size_step(
    state: WorkflowState,
    q_pixel_size: float,
    units: str = "A^-1",
    log: Callable[[str], None] | None = None,
):
    """Apply a Q pixel size directly to braggpeaks calibration."""

    braggpeaks = require_braggpeaks(state)
    value = float(q_pixel_size)
    if value <= 0:
        raise ValueError("Q pixel size must be positive.")
    braggpeaks.calibration.set_Q_pixel_size(value)
    braggpeaks.calibration.set_Q_pixel_units(units)
    braggpeaks.setcal()
    state.q_pixel_size = value
    state.q_pixel_units = units
    _log(log, f"Set Q pixel size: {value:g} {units}/px | calstate={getattr(braggpeaks, 'calstate', None)}")
    return braggpeaks


def calibrate_q_pixel_size_si_step(
    state: WorkflowState,
    px_guess: float,
    k_max: float = 1.0,
    bragg_k_power: float = 2.0,
    use_roi: bool = True,
    log: Callable[[str], None] | None = None,
) -> float:
    """Optional Si-based pixel-size calibration matching the notebook pattern."""

    import py4DSTEM

    braggpeaks = require_braggpeaks(state)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.0, 0.5, 0.5],
        [0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5],
        [0.75, 0.25, 0.75],
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.25],
    ], dtype=float)
    crystal = py4DSTEM.process.diffraction.Crystal(positions, 14, 5.431)
    crystal.calculate_structure_factors(float(k_max))

    bragg_use = _braggpeaks_for_roi(state, use_roi=use_roi)
    set_q_pixel_size_step(state, float(px_guess), log=log)
    if bragg_use is not braggpeaks:
        bragg_use.calibration.set_Q_pixel_size(float(px_guess))
        bragg_use.calibration.set_Q_pixel_units("A^-1")
        bragg_use.setcal()

    crystal.calibrate_pixel_size(
        bragg_peaks=bragg_use,
        bragg_k_power=float(bragg_k_power),
        k_max=float(k_max),
        set_calibration_in_place=True,
        verbose=True,
        plot_result=True,
    )

    px_fit = _get_q_pixel_size(bragg_use.calibration)
    set_q_pixel_size_step(state, px_fit, log=log)
    if bragg_use is not braggpeaks:
        bragg_use.calibration.set_Q_pixel_size(px_fit)
        bragg_use.calibration.set_Q_pixel_units("A^-1")
        bragg_use.setcal()
    _log(log, f"Calibrated Q pixel size: {px_fit:g} A^-1/px")
    return px_fit


def _make_crystal(material: str = "Si"):
    """
    Build a simple Crystal for pixel-size calibration overlays.

    Supported presets:
    - Si (diamond cubic), a=5.431 Å, Z=14
    - Au (fcc), a=4.0782 Å, Z=79
    """
    import py4DSTEM

    m = (material or "Si").strip().lower()
    if m in ("si", "silicon"):
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.5, 0.5],
            [0.25, 0.75, 0.75],
            [0.5, 0.0, 0.5],
            [0.75, 0.25, 0.75],
            [0.5, 0.5, 0.0],
            [0.75, 0.75, 0.25],
        ], dtype=float)
        return py4DSTEM.process.diffraction.Crystal(positions, 14, 5.431)
    if m in ("au", "gold"):
        # fcc conventional cell basis
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ], dtype=float)
        return py4DSTEM.process.diffraction.Crystal(positions, 79, 4.0782)
    raise ValueError(f"Unknown crystal preset: {material}. Use 'Si' or 'Au'.")


def _sync_q_pixel_to_objects(state: WorkflowState, bragg_use, px: float) -> None:
    braggpeaks = require_braggpeaks(state)
    braggpeaks.calibration.set_Q_pixel_size(float(px))
    braggpeaks.calibration.set_Q_pixel_units("A^-1")
    braggpeaks.setcal()
    if bragg_use is not braggpeaks:
        bragg_use.calibration.set_Q_pixel_size(float(px))
        bragg_use.calibration.set_Q_pixel_units("A^-1")
        bragg_use.setcal()
    state.q_pixel_size = float(px)
    state.q_pixel_units = "A^-1"


def q_pixel_overlay_figure(
    state: WorkflowState,
    px: float = 0.0137,
    k_max: float = 1.0,
    bragg_k_power: float = 2.0,
    use_roi: bool = True,
    log: Callable[[str], None] | None = None,
):
    """Notebook-style theory/experimental scattering overlay."""

    require_braggpeaks(state)
    if not _braggvectors_have_calibrated_origin(require_braggpeaks(state)):
        _ensure_braggpeaks_origin_for_strain(state, log=log)
    crystal = _make_crystal(getattr(state, "q_crystal", "Si"))
    crystal.calculate_structure_factors(float(k_max))
    bragg_use = _braggpeaks_for_roi(state, use_roi=use_roi)
    _sync_q_pixel_to_objects(state, bragg_use, float(px))

    # IMPORTANT: avoid plt.close('all') in Tk apps (can destroy TkAgg managers and
    # trigger "main thread is not in main loop" on Windows). Prefer drawing into
    # a standalone Figure so the GUI can embed it safely.
    from matplotlib.figure import Figure

    try:
        from .batch_figures import BATCH_Q_OVERLAY_SIZE, finalize_q_pixel_scattering_figure
    except ImportError:
        from batch_figures import BATCH_Q_OVERLAY_SIZE, finalize_q_pixel_scattering_figure

    fig = Figure(figsize=BATCH_Q_OVERLAY_SIZE)
    ax = fig.add_subplot(111)
    plotted = False
    kpow = float(bragg_k_power)
    # py4DSTEM.plot_scattering_intensity: experimental curve uses (k**bragg_k_power) on the
    # histogram, but the theory curve used k_power_scale=0 by default — peaks then look
    # systematically shifted when bragg_k_power != 0. Match notebook intent by applying the
    # same power to |g| on the structure-factor side. k_step=0.002 matches calibrate_pixel_size.
    plot_kwargs = dict(
        bragg_peaks=bragg_use,
        bragg_k_power=kpow,
        k_power_scale=kpow,
        k_max=float(k_max),
        k_step=0.002,
    )
    try:
        crystal.plot_scattering_intensity(
            figax=(fig, ax),
            **plot_kwargs,
        )
        plotted = True
    except TypeError:
        # Older py4DSTEM: no figax kwarg; still pass k alignment kwargs.
        try:
            crystal.plot_scattering_intensity(
                **plot_kwargs,
            )
            fig = plt.gcf()
            try:
                if fig.axes:
                    ax = fig.axes[0]
            except Exception:
                pass
            plotted = True
        except AssertionError as exc:
            raise RuntimeError(
                "Q pixel overlay requires Step 9 origin calibration on braggpeaks "
                "(center_guess + measure_origin / fit_origin). "
                f"py4DSTEM: {exc}"
            ) from exc
    except AssertionError as exc:
        raise RuntimeError(
            "Q pixel overlay requires Step 9 origin calibration on braggpeaks. "
            f"py4DSTEM: {exc}"
        ) from exc
    except Exception:
        plotted = False

    if plotted:
        try:
            ax.set_title(
                f"k_max={float(k_max):.2f} A^-1 | px={float(px):.7g} A^-1/px | "
                f"kpow={kpow:.2f} | ROI={bool(use_roi)}",
                fontsize=10,
            )
        except Exception:
            pass
        finalize_q_pixel_scattering_figure(fig)
    _log(log, f"Updated Q pixel overlay: px={float(px):.7g}, k_max={float(k_max):.2f}, kpow={kpow:.2f}")
    return fig


def finalize_q_pixel_refit_step(
    state: WorkflowState,
    px_guess: float = 0.0137,
    k_max: float = 1.0,
    bragg_k_power: float = 2.0,
    use_roi: bool = True,
    plot_result: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Notebook-style Finalize calibration (REFIT)."""

    require_braggpeaks(state)
    crystal = _make_crystal(getattr(state, "q_crystal", "Si"))
    crystal.calculate_structure_factors(float(k_max))
    bragg_use = _braggpeaks_for_roi(state, use_roi=use_roi)
    _sync_q_pixel_to_objects(state, bragg_use, float(px_guess))

    before = set(plt.get_fignums())
    crystal.calibrate_pixel_size(
        bragg_peaks=bragg_use,
        bragg_k_power=float(bragg_k_power),
        k_max=float(k_max),
        set_calibration_in_place=True,
        verbose=False,
        plot_result=bool(plot_result),
    )
    after = set(plt.get_fignums())
    new_nums = sorted(after - before)
    if bool(plot_result):
        fig = plt.figure(new_nums[-1]) if new_nums else plt.gcf()
    else:
        from matplotlib.figure import Figure

        fig = Figure(figsize=(7.2, 5.4))
        ax = fig.add_subplot(111)
        ax.axis("off")
    px_fit = _get_q_pixel_size(bragg_use.calibration)
    _sync_q_pixel_to_objects(state, bragg_use, px_fit)
    _annotate_q_pixel_fit(fig, px_guess, px_fit, use_roi, bragg_k_power, k_max)
    try:
        from .batch_figures import finalize_q_pixel_scattering_figure
    except ImportError:
        from batch_figures import finalize_q_pixel_scattering_figure

    try:
        finalize_q_pixel_scattering_figure(fig)
    except Exception:
        pass
    _log(log, f"Q pixel REFIT complete: guess={float(px_guess):.10g}, fit={float(px_fit):.10g}")
    return {"px_fit": float(px_fit), "figure": fig}


def test_q_pixel_size_step(
    state: WorkflowState,
    px0: float = 0.0137,
    test_step: float = 1e-4,
    n_figures: int = 7,
    k_max: float = 1.0,
    bragg_k_power: float = 2.0,
    use_roi: bool = True,
    show_each_fit_plot: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run an odd-numbered sweep around px0 and keep fit figures plus summary."""

    require_braggpeaks(state)
    n = int(n_figures)
    if n not in (3, 5, 7):
        raise ValueError("n_figures must be 3, 5, or 7.")
    half = n // 2
    offsets = np.arange(-half, half + 1, dtype=float) * float(test_step)

    crystal = _make_crystal(getattr(state, "q_crystal", "Si"))
    crystal.calculate_structure_factors(float(k_max))
    bragg_use = _braggpeaks_for_roi(state, use_roi=use_roi)

    px_backup = _get_q_pixel_size(require_braggpeaks(state).calibration)
    px_guess_list: list[float] = []
    px_fit_list: list[float] = []
    figures = []
    failures = []

    for dpx in offsets:
        px_guess = float(px0) + float(dpx)
        _sync_q_pixel_to_objects(state, bragg_use, px_guess)
        before = set(plt.get_fignums())
        try:
            crystal.calibrate_pixel_size(
                bragg_peaks=bragg_use,
                bragg_k_power=float(bragg_k_power),
                k_max=float(k_max),
                set_calibration_in_place=True,
                verbose=False,
                plot_result=bool(show_each_fit_plot),
            )
        except Exception as exc:
            failures.append((px_guess, str(exc)))
            _log(log, f"Q pixel test failed for guess={px_guess:.10g}: {exc}")
            continue
        after = set(plt.get_fignums())
        new_nums = sorted(after - before)
        fig = None
        if bool(show_each_fit_plot):
            fig = plt.figure(new_nums[-1]) if new_nums else plt.gcf()
        px_fit = _get_q_pixel_size(bragg_use.calibration)
        _sync_q_pixel_to_objects(state, bragg_use, px_fit)
        if fig is not None:
            _annotate_q_pixel_fit(fig, px_guess, px_fit, use_roi, bragg_k_power, k_max)
            try:
                fig.axes[0].set_title(f"Pixel-size fit | guess={px_guess:.7g} fit={px_fit:.7g}")
            except Exception:
                pass
            figures.append(fig)
        px_guess_list.append(px_guess)
        px_fit_list.append(float(px_fit))
        _log(log, f"guess px={px_guess:.10g} -> fitted px={float(px_fit):.10g} (delta={float(px_fit)-px_guess:+.2e})")

    if np.isfinite(px_backup):
        _sync_q_pixel_to_objects(state, bragg_use, float(px_backup))

    summary_fig = None
    if len(px_fit_list) >= 2:
        summary_fig = _q_pixel_summary_figure(np.asarray(px_guess_list), np.asarray(px_fit_list), float(px0), float(test_step))
        figures.append(summary_fig)
    _log(log, f"Q pixel test complete: {len(px_fit_list)} successful fit(s), {len(failures)} failure(s).")
    return {
        "figures": figures,
        "summary_figure": summary_fig,
        "px_guess": np.asarray(px_guess_list),
        "px_fit": np.asarray(px_fit_list),
        "failures": failures,
    }


def _annotate_q_pixel_fit(fig, px_guess, px_fit, use_roi, bragg_k_power, k_max) -> None:
    try:
        ax = fig.axes[0]
        text = (
            f"px_guess = {float(px_guess):.10g} A^-1/px\n"
            f"px_fit   = {float(px_fit):.10g} A^-1/px\n"
            f"delta    = {float(px_fit) - float(px_guess):+.2e}\n"
            f"ROI={bool(use_roi)} | kpow={float(bragg_k_power):.2f} | kmax={float(k_max):.2f}"
        )
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", alpha=0.25),
        )
    except Exception:
        pass


def _q_pixel_summary_figure(px_guess, px_fit, px0: float, test_step: float):
    residual = px_fit - px_guess
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    lo = float(min(px_guess.min(), px_fit.min()))
    hi = float(max(px_guess.max(), px_fit.max()))
    pad = max((hi - lo) * 0.2, 5e-6)
    axes[0].plot(px_guess, px_fit, marker="o", linestyle="-", label="px_fit")
    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", label="y=x")
    axes[0].set_xlabel("px guess (A^-1/px)")
    axes[0].set_ylabel("px fitted (A^-1/px)")
    axes[0].set_title(f"Guess vs fitted (N={len(px_fit)}, step={test_step:.1e})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(px_guess - float(px0), residual, marker="o", linestyle="-")
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_xlabel("offset dpx")
    axes[1].set_ylabel("px_fit - px_guess")
    axes[1].set_title("Sensitivity residual")
    axes[1].grid(True, alpha=0.3)
    return fig


def _get_q_pixel_size(calibration) -> float:
    if hasattr(calibration, "get_Q_pixel_size"):
        try:
            return float(calibration.get_Q_pixel_size())
        except Exception:
            pass
    for attr in ("Q_pixel_size", "q_pixel_size"):
        if hasattr(calibration, attr):
            try:
                return float(getattr(calibration, attr))
            except Exception:
                pass
    return float("nan")


def fit_q_pixel_size_threadsafe(
    state: WorkflowState,
    px_guess: float = 0.0137,
    k_max: float = 1.0,
    bragg_k_power: float = 2.0,
    use_roi: bool = True,
    log: Callable[[str], None] | None = None,
) -> float:
    """Fit Q-pixel size using ``crystal.calibrate_pixel_size`` **without** any
    pyplot interaction.

    Safe to call from worker threads with a TkAgg matplotlib backend.
    Unlike :func:`finalize_q_pixel_refit_step`, this function never calls
    ``plt.get_fignums()`` or any other pyplot API, so it cannot trigger the
    ``Tcl_AsyncDelete: async handler deleted by the wrong thread`` crash that
    occurs when pyplot touches the Tk event loop from a non-main thread.

    Returns the fitted pixel size (Å⁻¹/px).  The calibration is applied
    in-place on *state.braggpeaks* via ``_sync_q_pixel_to_objects``.
    """
    require_braggpeaks(state)
    crystal = _make_crystal(getattr(state, "q_crystal", "Si"))
    crystal.calculate_structure_factors(float(k_max))
    bragg_use = _braggpeaks_for_roi(state, use_roi=use_roi)
    _sync_q_pixel_to_objects(state, bragg_use, float(px_guess))

    crystal.calibrate_pixel_size(
        bragg_peaks=bragg_use,
        bragg_k_power=float(bragg_k_power),
        k_max=float(k_max),
        set_calibration_in_place=True,
        verbose=False,
        plot_result=False,   # critical: no pyplot interaction
    )

    px_fit = _get_q_pixel_size(bragg_use.calibration)
    if not (0 < px_fit < 1.0):          # sanity-check: fall back to guess
        _log(log, f"fit_q_pixel_size_threadsafe: implausible fit result "
                  f"{px_fit:.6g} — reverting to guess {px_guess:.6g}")
        px_fit = float(px_guess)

    _sync_q_pixel_to_objects(state, bragg_use, px_fit)
    _log(log, f"fit_q_pixel_size_threadsafe: guess={px_guess:.7g} fit={px_fit:.10g}")
    return px_fit


def update_strain_basis_params(
    state: WorkflowState,
    min_spacing: int = 5,
    min_absolute_intensity: int = 80,
    max_num_peaks: int = 60,
    edge_boundary: int = 4,
    vmin: float = 0.0,
    vmax: float = 0.995,
    qr_rotation: float = 0.0,
    qr_flip: bool = False,
    manual_enabled: bool = False,
    index_origin: int = 0,
    index_g1: int = 0,
    index_g2: int = 0,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    braggpeaks = require_braggpeaks(state)
    cbv: dict[str, Any] = {
        "minSpacing": int(min_spacing),
        "minAbsoluteIntensity": int(min_absolute_intensity),
        "maxNumPeaks": int(max_num_peaks),
        "edgeBoundary": int(edge_boundary),
        "vis_params": {"vmin": float(vmin), "vmax": float(vmax)},
    }
    if bool(manual_enabled):
        # Notebook supports manual indices for choose_basis_vectors
        cbv.update(
            {
                "index_origin": int(index_origin),
                "index_g1": int(index_g1),
                "index_g2": int(index_g2),
            }
        )
    state.strain_basis_params = {
        "choose_basis_vectors": cbv,
        "qr_rotation": float(qr_rotation),
        "qr_flip": bool(qr_flip),
        "manual_enabled": bool(manual_enabled),
    }
    try:
        # Notebook uses degrees. Prefer the *_degrees method if available.
        if hasattr(braggpeaks.calibration, "set_QR_rotation_degrees"):
            braggpeaks.calibration.set_QR_rotation_degrees(float(qr_rotation))
        else:
            braggpeaks.calibration.set_QR_rotation(float(qr_rotation))
        braggpeaks.calibration.set_QR_flip(bool(qr_flip))
        braggpeaks.setcal()
    except Exception as exc:
        _log(log, f"Warning: could not set QR calibration directly: {exc}")
    _log(log, f"Updated strain basis params. QR_rotation={qr_rotation}, QR_flip={qr_flip}")
    return state.strain_basis_params


def apply_import_params_to_workflow_state(
    state: WorkflowState,
    obj: dict[str, Any],
    log: Callable[[str], None] | None = None,
) -> None:
    """
    Apply a normalized calibration dict (e.g. :func:`calib_params_io.load_params_dict`)
    to ``state`` without touching Tk widgets. Requires ``braggpeaks`` for Q pixel / basis
    updates that touch calibration objects; other fields apply as soon as data exist.
    """
    if not isinstance(obj, dict):
        return

    org = (obj.get("origin") or {}) or {}
    cg = org.get("center_guess")
    if cg is not None:
        try:
            if isinstance(cg, (list, tuple)) and len(cg) == 2:
                y0, x0 = int(cg[0]), int(cg[1])
                samp = int(obj.get("sampling", state.origin_sampling) or 1)
                state.origin_sampling = max(1, samp)
                set_origin_center_guess(state, (y0, x0), sampling=state.origin_sampling, log=log)
        except Exception as exc:
            _log(log, f"Note: could not set origin from params: {exc}")
    elif obj.get("sampling", None) is not None:
        try:
            state.origin_sampling = max(1, int(obj["sampling"]))
        except Exception:
            pass

    px = (obj.get("pixel_size") or {}) or {}
    qpx = None
    if px.get("pixel_size_ui_value", None) is not None:
        try:
            qpx = float(px["pixel_size_ui_value"])
        except (TypeError, ValueError):
            qpx = None
    if qpx is None and obj.get("q_pixel_size", None) is not None:
        try:
            qpx = float(obj["q_pixel_size"])
        except (TypeError, ValueError):
            qpx = None
    units = str(px.get("Q_pixel_units") or obj.get("q_pixel_units") or state.q_pixel_units or "A^-1")
    if qpx is not None and qpx == qpx and qpx > 0:
        try:
            if state.braggpeaks is not None:
                set_q_pixel_size_step(state, float(qpx), units=units, log=log)
            else:
                state.q_pixel_size = float(qpx)
                state.q_pixel_units = units
        except Exception as exc:
            _log(log, f"Note: could not set Q pixel size: {exc}")

    dp = obj.get("detect_params", {}) or {}
    if isinstance(dp, dict) and dp:
        merged = copy.deepcopy(state.detect_params)
        merged.update({k: v for k, v in dp.items() if v is not None})
        state.detect_params = merged

    dprev = obj.get("disk_preview_params", {}) or {}
    if isinstance(dprev, dict) and dprev:
        merged = copy.deepcopy(state.disk_preview_params)
        merged.update({k: v for k, v in dprev.items() if v is not None})
        state.disk_preview_params = merged

    sbp: dict[str, Any] | None = None
    top_sbp = obj.get("strain_basis_params")
    if isinstance(top_sbp, dict) and isinstance(top_sbp.get("choose_basis_vectors"), dict):
        sbp = copy.deepcopy(top_sbp)
    else:
        sb = obj.get("strain_basis", {}) or {}
        base = sb.get("STRAIN_BASIS_PARAMS", {}) or {}
        eff = sb.get("choose_basis_kwargs_effective", {}) or {}
        pick_pre = sb.get("STRAIN_PICK", {}) or {}
        cb = (
            (base.get("choose_basis_vectors", {}) or eff)
            or ((pick_pre.get("choose_basis_kwargs", {}) or {}) if pick_pre else {})
            or {}
        )
        if isinstance(cb, dict) and cb:
            sbp = {
                "choose_basis_vectors": copy.deepcopy(cb),
                "qr_rotation": float(obj.get("qr_rotation", 0.0) or 0.0),
                "qr_flip": bool(obj.get("qr_flip", False)),
                "manual_enabled": False,
            }
            pick = pick_pre or (sb.get("STRAIN_PICK", {}) or {})
            qr = pick.get("QR_rotation", None)
            if qr is None:
                rotb = obj.get("rotation", {}) or {}
                qr = rotb.get("QR_rotation_calibration", None)
            if qr is not None:
                sbp["qr_rotation"] = float(qr)
            if pick.get("QR_flip", None) is not None:
                sbp["qr_flip"] = bool(pick["QR_flip"])
            elif (obj.get("rotation", {}) or {}).get("QR_flip", None) is not None:
                sbp["qr_flip"] = bool((obj.get("rotation", {}) or {})["QR_flip"])
            me = pick.get("manual_enabled", None)
            if me is None:
                me = sb.get("manual_pick", None)
            if me is not None:
                sbp["manual_enabled"] = bool(me)
            elif isinstance(cb, dict) and all(
                k in cb and cb.get(k) is not None for k in ("index_origin", "index_g1", "index_g2")
            ):
                sbp["manual_enabled"] = True
            mi = pick.get("manual_indices", None)
            if not isinstance(mi, (list, tuple)) or len(mi) != 3:
                mi = (sb.get("index_origin", None), sb.get("index_g1", None), sb.get("index_g2", None))
            if (
                (not isinstance(mi, (list, tuple)) or len(mi) != 3 or any(x is None for x in mi))
                and isinstance(cb, dict)
                and cb.get("index_origin", None) is not None
                and cb.get("index_g1", None) is not None
                and cb.get("index_g2", None) is not None
            ):
                mi = (cb.get("index_origin"), cb.get("index_g1"), cb.get("index_g2"))
            if (
                isinstance(mi, (list, tuple))
                and len(mi) == 3
                and mi[0] is not None
                and mi[1] is not None
                and mi[2] is not None
            ):
                sbp["choose_basis_vectors"]["index_origin"] = int(mi[0])
                sbp["choose_basis_vectors"]["index_g1"] = int(mi[1])
                sbp["choose_basis_vectors"]["index_g2"] = int(mi[2])

    if isinstance(sbp, dict) and isinstance(sbp.get("choose_basis_vectors"), dict):
        cbv = sbp["choose_basis_vectors"]
        vp = (cbv.get("vis_params") or {}) if isinstance(cbv, dict) else {}
        try:
            if state.braggpeaks is not None:
                update_strain_basis_params(
                    state,
                    min_spacing=int(cbv.get("minSpacing", 5)),
                    min_absolute_intensity=int(cbv.get("minAbsoluteIntensity", 80)),
                    max_num_peaks=int(cbv.get("maxNumPeaks", 60)),
                    edge_boundary=int(cbv.get("edgeBoundary", 4)),
                    vmin=float(vp.get("vmin", 0.0)),
                    vmax=float(vp.get("vmax", 0.995)),
                    qr_rotation=float(sbp.get("qr_rotation", 0.0)),
                    qr_flip=bool(sbp.get("qr_flip", False)),
                    manual_enabled=bool(sbp.get("manual_enabled", False)),
                    index_origin=int(cbv.get("index_origin", 0)),
                    index_g1=int(cbv.get("index_g1", 0)),
                    index_g2=int(cbv.get("index_g2", 0)),
                    log=log,
                )
            else:
                state.strain_basis_params = copy.deepcopy(sbp)
        except Exception as exc:
            _log(log, f"Note: could not apply strain basis params to braggpeaks: {exc}")
            state.strain_basis_params = copy.deepcopy(sbp)

    gs0 = (state.strain_params or {}).get("get_strain") or {}
    coord_rot = float(gs0.get("coordinate_rotation", 90.0))
    layout = str(gs0.get("layout", "horizontal"))
    mps0 = (state.strain_params or {}).get("set_max_peak_spacing") or {}
    max_spacing = float(mps0.get("max_peak_spacing", 2.0))
    vr = gs0.get("vrange", [-2.0, 2.0])
    vt = gs0.get("vrange_theta", [-45.0, 45.0])
    try:
        vrange = (float(vr[0]), float(vr[1]))
    except Exception:
        vrange = (-2.0, 2.0)
    try:
        vtheta = (float(vt[0]), float(vt[1]))
    except Exception:
        vtheta = (-45.0, 45.0)

    sset = (obj.get("strain_settings") or {}) or {}
    sp = obj.get("strain_params")
    if isinstance(sp, dict):
        gs = (sp.get("get_strain") or {}) or {}
        if gs.get("coordinate_rotation", None) is not None:
            coord_rot = float(gs["coordinate_rotation"])
        if gs.get("layout", None) is not None:
            layout = str(gs["layout"])
        vr2 = gs.get("vrange", None)
        if vr2 is not None:
            a = np.asarray(vr2, dtype=float).ravel()
            if a.size == 2 and np.isfinite(a).all():
                vrange = (float(a[0]), float(a[1]))
        vt2 = gs.get("vrange_theta", None)
        if vt2 is not None:
            a = np.asarray(vt2, dtype=float).ravel()
            if a.size == 2 and np.isfinite(a).all():
                vtheta = (float(a[0]), float(a[1]))
        mps = (sp.get("set_max_peak_spacing") or {}) or {}
        if mps.get("max_peak_spacing", None) is not None:
            max_spacing = float(mps["max_peak_spacing"])
    if sset.get("coordinate_rotation", None) is not None:
        coord_rot = float(sset["coordinate_rotation"])
    if sset.get("layout", None) is not None:
        layout = str(sset["layout"])
    if sset.get("max_peak_spacing", None) is not None:
        max_spacing = float(sset["max_peak_spacing"])
    vr = sset.get("vrange", None)
    if vr is not None:
        try:
            a = np.asarray(vr, dtype=float).ravel()
            if a.size == 2 and np.isfinite(a).all():
                vrange = (float(a[0]), float(a[1]))
        except Exception:
            pass
    vrt = sset.get("vrange_theta", None)
    if vrt is not None:
        try:
            a = np.asarray(vrt, dtype=float).ravel()
            if a.size == 2 and np.isfinite(a).all():
                vtheta = (float(a[0]), float(a[1]))
        except Exception:
            pass

    try:
        update_strain_params(
            state,
            coordinate_rotation=coord_rot,
            max_peak_spacing=max_spacing,
            layout=str(layout or "horizontal"),
            vrange=vrange,
            vrange_theta=vtheta,
            log=log,
        )
    except Exception as exc:
        _log(log, f"Note: could not update strain_params: {exc}")

    ref = obj.get("reference_disks", {}) or {}
    rxs, rys = ref.get("rxs"), ref.get("rys")
    if (
        state.braggpeaks is not None
        and isinstance(rxs, (list, tuple))
        and isinstance(rys, (list, tuple))
        and len(rxs) == 6
        and len(rys) == 6
    ):
        try:
            pts = [(float(x), float(y)) for x, y in zip(rxs, rys)]
            set_bragg_points(state, pts, log=log)
        except Exception as exc:
            _log(log, f"Note: could not set reference_disks: {exc}")

    try:
        roi = obj.get("ROI", {}) or {}
        if state.datacube is not None and all(
            roi.get(k) is not None for k in ("xMin", "xMax", "yMin", "yMax")
        ):
            x0, x1 = int(roi["xMin"]), int(roi["xMax"])
            y0, y1 = int(roi["yMin"]), int(roi["yMax"])
            set_roi_from_bounds(state, (x0, x1, y0, y1), log=log)
    except Exception as exc:
        _log(log, f"Note: could not set ROI from params: {exc}")

    _log(log, "Applied calibration dict to workflow state (reference / batch path).")


def setup_basis_step(state: WorkflowState, log: Callable[[str], None] | None = None):
    """Build a StrainMap and choose basis vectors using current basis params."""

    import py4DSTEM

    braggpeaks = require_braggpeaks(state)
    strainmap = py4DSTEM.StrainMap(braggvectors=braggpeaks)
    _choose_basis_vectors_safely(strainmap, state.strain_basis_params["choose_basis_vectors"])
    _apply_manual_basis_pick(state, strainmap)
    state.strainmap_full = strainmap
    _log(log, "Interactive basis calibration setup complete.")
    return strainmap


def preview_basis_figure_step(
    state: WorkflowState,
    log: Callable[[str], None] | None = None,
    *,
    scan_label: str | None = None,
):
    """Run choose_basis_vectors and capture the produced figure (for live preview)."""

    import py4DSTEM

    braggpeaks = require_braggpeaks(state)
    before = set(plt.get_fignums())
    strainmap = py4DSTEM.StrainMap(braggvectors=braggpeaks)
    # For preview we want the original (3-panel) choose_basis_vectors figures.
    strainmap.choose_basis_vectors(**state.strain_basis_params["choose_basis_vectors"])
    after = set(plt.get_fignums())
    new_nums = sorted(after - before)
    figs = [plt.figure(n) for n in new_nums] if new_nums else [plt.gcf()]

    pick = _apply_manual_basis_pick(state, strainmap)
    label = (scan_label or "").strip()
    for i, f in enumerate(figs, start=1):
        _annotate_basis_pick_on_figure(f, pick.get("g1"), pick.get("g2"), pick.get("info", ""))
        if label:
            sub = f"Strain basis (panel {i})" if len(figs) > 1 else "Strain basis"
            stamp_figure_scan_title(f, label, sub, y=1.0 if len(getattr(f, "axes", [])) >= 2 else 0.98)
    state.strainmap_full = strainmap
    try:
        state.basis_preview_figures = list(figs)
    except Exception:
        state.basis_preview_figures = []
    _log(log, "Basis preview updated (choose_basis_vectors).")
    return {"strainmap": strainmap, "figures": figs, "manual_pick_info": pick.get("info", "Manual pick: off")}


def _choose_basis_vectors_safely(strainmap, params: dict[str, Any]) -> None:
    """
    Run choose_basis_vectors without forcing pyplot figure creation.
    Some py4DSTEM versions support plot=False; try it first.
    """

    try:
        strainmap.choose_basis_vectors(**params, plot=False)
        return
    except TypeError:
        pass
    try:
        strainmap.choose_basis_vectors(**params, plot_result=False)
        return
    except TypeError:
        pass
    # Fallback: default behavior
    strainmap.choose_basis_vectors(**params)


def _apply_manual_basis_pick(state: WorkflowState, strainmap) -> dict[str, Any]:
    """
    Optional: pick which g1/g2 rows to use (when multiple candidates exist).
    This mirrors notebook patterns where the first match `[0]` is used; here
    you can pick `[k]` among matches.
    """

    cfg = state.strain_basis_params.get("manual_pick", {}) if isinstance(state.strain_basis_params, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False, "info": "Manual pick: off", "g1": None, "g2": None}

    g1_pick = int(cfg.get("g1_pick", 0))
    g2_pick = int(cfg.get("g2_pick", 0))

    bd = getattr(strainmap, "braggdirections", None)
    rec = None
    if bd is not None:
        rec = getattr(bd, "data", bd)
    rec = np.asarray(rec) if rec is not None else None
    if rec is None or rec.size == 0:
        return {"enabled": True, "info": "Manual pick: enabled but braggdirections unavailable", "g1": None, "g2": None}

    try:
        mask_g1 = (np.abs(rec["g1_ind"]) == 1) & (rec["g2_ind"] == 0)
        mask_g2 = (rec["g1_ind"] == 0) & (np.abs(rec["g2_ind"]) == 1)
        g1_rows = rec[mask_g1]
        g2_rows = rec[mask_g2]
        if len(g1_rows) == 0 or len(g2_rows) == 0:
            return {"enabled": True, "info": "Manual pick: could not locate g1/g2 candidates", "g1": None, "g2": None}
        if g1_pick < 0 or g1_pick >= len(g1_rows) or g2_pick < 0 or g2_pick >= len(g2_rows):
            return {
                "enabled": True,
                "info": f"Manual pick: invalid (g1 0..{len(g1_rows)-1}, g2 0..{len(g2_rows)-1})",
                "g1": None,
                "g2": None,
            }

        g1 = np.array([float(g1_rows[g1_pick]["qx"]), float(g1_rows[g1_pick]["qy"])], dtype=float)
        g2 = np.array([float(g2_rows[g2_pick]["qx"]), float(g2_rows[g2_pick]["qy"])], dtype=float)

        # Best-effort: set on strainmap if the attribute exists.
        for a, v in (("g1_exp", g1), ("g2_exp", g2), ("g1", g1), ("g2", g2)):
            if hasattr(strainmap, a):
                try:
                    setattr(strainmap, a, v)
                except Exception:
                    pass

        info = (
            f"Manual pick ON | g1_pick={g1_pick}/{len(g1_rows)-1} g2_pick={g2_pick}/{len(g2_rows)-1} | "
            f"g1=({g1[0]:.4g},{g1[1]:.4g}) g2=({g2[0]:.4g},{g2[1]:.4g})"
        )
        state.strain_basis_params["manual_pick_result"] = info
        return {"enabled": True, "info": info, "g1": g1, "g2": g2}
    except Exception as exc:
        return {"enabled": True, "info": f"Manual pick: failed ({exc})", "g1": None, "g2": None}


def _annotate_basis_pick_on_figure(fig, g1=None, g2=None, info: str = "") -> None:
    """
    Overlay the manually-picked basis vectors on top of the original
    choose_basis_vectors() figure, without changing its layout.
    """

    if fig is None:
        return
    if g1 is None and g2 is None:
        return

    # Heuristic: annotate the first axis that looks like a qx/qy scatter plot.
    ax_target = None
    for ax in getattr(fig, "axes", []):
        try:
            xl = (ax.get_xlabel() or "").lower()
            yl = (ax.get_ylabel() or "").lower()
            if ("qx" in xl and "qy" in yl) or ("qx" in xl) or ("qy" in yl):
                ax_target = ax
                break
        except Exception:
            continue
    if ax_target is None and getattr(fig, "axes", None):
        ax_target = fig.axes[0]

    ax = ax_target
    try:
        ax.scatter([0], [0], s=20, c="k", zorder=5)
    except Exception:
        pass
    try:
        if g1 is not None:
            ax.plot([0, float(g1[0])], [0, float(g1[1])], color="tab:red", linewidth=2, zorder=6)
            ax.scatter([float(g1[0])], [float(g1[1])], s=40, c="tab:red", zorder=7)
            ax.text(float(g1[0]), float(g1[1]), " g1", color="tab:red")
        if g2 is not None:
            ax.plot([0, float(g2[0])], [0, float(g2[1])], color="tab:blue", linewidth=2, zorder=6)
            ax.scatter([float(g2[0])], [float(g2[1])], s=40, c="tab:blue", zorder=7)
            ax.text(float(g2[0]), float(g2[1]), " g2", color="tab:blue")
    except Exception:
        pass

    if info:
        try:
            ax.text(
                0.02,
                0.98,
                info,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=12,
                bbox=dict(boxstyle="round", alpha=0.25),
            )
        except Exception:
            pass


def update_strain_params(
    state: WorkflowState,
    coordinate_rotation: float = 90.0,
    max_peak_spacing: float = 2.0,
    layout: str = "horizontal",
    vrange=(-2, 2),
    vrange_theta=(-45.0, 45.0),
    cmap: str = "RdBu_r",
    cmap_theta: str = "PRGn",
    show_orientation: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    state.strain_params = {
        "set_max_peak_spacing": {"max_peak_spacing": float(max_peak_spacing)},
        "fit_basis_vectors": {},
        "get_strain": {
            "coordinate_rotation": float(coordinate_rotation),
            "layout": str(layout),
            "vrange": list(vrange),
            "vrange_theta": list(vrange_theta),
            "cmap": str(cmap),
            "cmap_theta": str(cmap_theta),
        },
        "show_orientation": bool(show_orientation),  # Fast4D-only; not a py4DSTEM kwarg
    }
    _log(log, f"Updated strain params: {state.strain_params}")
    return state.strain_params


def _heatmap_on_ax_matches_reference_array(ax: Any, ref: np.ndarray | None) -> bool:
    """True if a large heatmap on ``ax`` matches ``ref`` (same shape, nearly identical values)."""
    if ref is None or getattr(ref, "ndim", 0) != 2:
        return False
    ref = np.asarray(ref, dtype=np.float64)
    for mappable, _ in _iter_strain_heatmap_mappables(ax):
        try:
            arr = np.asarray(mappable.get_array(), dtype=np.float64)
        except Exception:
            continue
        if arr.size != ref.size:
            continue
        if arr.shape != ref.shape:
            try:
                arr = arr.reshape(ref.shape)
            except Exception:
                continue
        ok = np.isfinite(arr) & np.isfinite(ref)
        if int(np.count_nonzero(ok)) < 64:
            continue
        dv = np.abs(arr[ok] - ref[ok])
        scale = float(np.nanmax(np.abs(ref[ok])) or 0.0)
        tol = max(1e-5, 1e-6 * max(scale, 1e-9))
        err = float(np.nanmax(dv))
        if err <= tol:
            return True
    return False


def _axis_title_suggests_theta(ax: Any) -> bool:
    try:
        blob = " ".join(
            [
                str(ax.get_title() or ""),
                str(ax.get_ylabel() or ""),
                str(ax.get_xlabel() or ""),
            ]
        ).lower()
    except Exception:
        return False
    if "theta" in blob or "θ" in blob:
        return True
    if "orientation" in blob:
        return True
    if "rotation" in blob and "strain" not in blob:
        return True
    return False


def _strain_colorbar_labels_fraction_as_percent(cb: Any) -> bool:
    """
    py4DSTEM strain maps are usually in **fraction** while the GUI ``vrange`` is in **%**.
    In that case clims are set to ``vrange/100`` but the colorbar axis is still labeled ``%``;
    tick values must be multiplied by 100 to match the GUI.
    """
    if _strain_colorbar_associates_theta(cb):
        return False
    data_ax = getattr(cb.mappable, "axes", None)
    if data_ax is not None and _axis_title_suggests_theta(data_ax):
        return False
    try:
        lab = (str(cb.ax.get_xlabel() or "") + str(cb.ax.get_ylabel() or "")).lower()
    except Exception:
        lab = ""
    if "%" not in lab and "percent" not in lab:
        return False
    try:
        lo, hi = (float(x) for x in cb.mappable.get_clim())
    except Exception:
        return False
    span = max(abs(lo), abs(hi))
    return 0 < span <= 2.0


def _strain_colorbar_associates_theta(cb: Any) -> bool:
    """True if this colorbar belongs to the θ / orientation panel (layout varies by py4DSTEM)."""
    data_ax = getattr(cb.mappable, "axes", None)
    if data_ax is not None and _axis_title_suggests_theta(data_ax):
        return True
    try:
        xs = str(cb.ax.get_xlabel() or "")
        ys = str(cb.ax.get_ylabel() or "")
        blob = (xs + ys).lower()
        if "theta" in blob or "θ" in blob:
            return True
        if "deg" in blob or "°" in (xs + ys):
            return True
        if "orient" in blob:
            return True
    except Exception:
        pass
    return False


def _reference_strain_scale_exx_fractional(raw_tensor) -> float:
    """Typical |ε_xx| (fraction) from strainmap_g1g2 raw for percent vs fraction clim choice."""
    if raw_tensor is None:
        return 0.0
    try:
        S = np.squeeze(np.asarray(raw_tensor))
        if S.ndim != 3:
            return 0.0
        exx = np.asarray(_strain_channel(S, 1), dtype=float)
        return float(np.nanmax(np.abs(exx)) or 0.0)
    except Exception:
        return 0.0


def _effective_orientation_clims(
    tmin_user: float,
    tmax_user: float,
    raw_tensor: Any,
) -> tuple[float, float]:
    """
    Map GUI ``vrange_theta`` (always **degrees** — matches the ``theta (deg)`` field in
    ``app_tk`` and the ``state.py`` defaults of ``[-45.0, 45.0]``) to the actual clim
    space of the orientation mappable.

    The stored θ array can live in either unit depending on the py4DSTEM version:
      * radians when ``max|θ| ≲ π``  →  return ``deg2rad(t_lo), deg2rad(t_hi)``
      * degrees otherwise            →  return ``(t_lo, t_hi)`` unchanged

    The companion :func:`_refresh_strain_figure_colorbars` always renders tick labels
    in degrees with a ``°`` symbol so the *colorbar text* is consistent with the GUI
    label regardless of which unit the mappable stores internally.
    """
    t_lo, t_hi = float(tmin_user), float(tmax_user)
    maps = _strain_maps_dict_from_raw(raw_tensor)
    th = maps.get("theta")
    if th is None:
        return t_lo, t_hi
    a = np.asarray(th, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return t_lo, t_hi

    p99_abs = float(np.percentile(np.abs(a), 99))
    mx_abs = float(np.nanmax(np.abs(a)))

    data_is_radians = mx_abs <= float(np.pi) + 0.08 and p99_abs <= float(np.pi) + 0.08
    if data_is_radians:
        return float(np.deg2rad(t_lo)), float(np.deg2rad(t_hi))

    # Data is already in degrees: widen tiny symmetric legacy ranges (e.g. ±1°) so
    # the colorbar scale still spans the actual data when users haven't customised.
    gui_half = max(abs(t_lo), abs(t_hi))
    sym = np.isclose(t_lo + t_hi, 0.0, rtol=1e-9, atol=1e-9)
    if sym and gui_half <= 2.5 and p99_abs > max(4.5, 1.55 * gui_half):
        cap = float(min(max(p99_abs * 1.06, 5.0), 180.0))
        return (-cap, cap)
    return t_lo, t_hi


def _strain_panel_clims(
    vmin_p: float,
    vmax_p: float,
    tmin: float,
    tmax: float,
    *,
    theta_ax: bool,
    amax_display: float,
    ref_frac: float,
) -> tuple[float, float] | None:
    if theta_ax:
        return (tmin, tmax)
    if ref_frac > 1e-12 and np.isfinite(amax_display) and amax_display > 0:
        ratio = float(amax_display) / float(ref_frac)
        if ratio > 25.0:
            return (vmin_p, vmax_p)
        if ratio < 8.0:
            return (vmin_p / 100.0, vmax_p / 100.0)
    if amax_display > 1.0:
        return (vmin_p, vmax_p)
    return (vmin_p / 100.0, vmax_p / 100.0)


def _iter_strain_heatmap_mappables(ax: Any):
    """py4DSTEM may use imshow (images) or pcolormesh/quadmesh (collections)."""
    for im in getattr(ax, "images", ()) or ():
        try:
            arr = np.asarray(im.get_array(), dtype=float)
        except Exception:
            continue
        if arr.ndim != 2 or min(arr.shape) < 3:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size < 16:
            continue
        yield im, float(np.nanmax(np.abs(finite)) or 0.0)

    for coll in getattr(ax, "collections", ()) or ():
        if not hasattr(coll, "set_clim") or not hasattr(coll, "get_array"):
            continue
        try:
            arr = coll.get_array()
        except Exception:
            continue
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float).ravel()
        finite = a[np.isfinite(a)]
        if finite.size < 16:
            continue
        yield coll, float(np.nanmax(np.abs(finite)) or 0.0)


def _iter_figure_matplotlib_colorbars(fig: Any):
    """Yield unique ``matplotlib.colorbar.Colorbar`` instances (``findobj`` misses some mpl layouts)."""
    import matplotlib.colorbar as cbar_module

    seen: set[int] = set()

    def _yield(cb: Any) -> Any:
        if cb is None or not isinstance(cb, cbar_module.Colorbar):
            return
        i = id(cb)
        if i in seen:
            return
        seen.add(i)
        return cb

    try:
        for cb in list(getattr(fig, "colorbars", ())):
            out = _yield(cb)
            if out is not None:
                yield out
    except Exception:
        pass

    try:
        for cb in fig.findobj(lambda obj: isinstance(obj, cbar_module.Colorbar)):
            out = _yield(cb)
            if out is not None:
                yield out
    except Exception:
        pass

    try:
        for ax in fig.axes:
            for artist in list(getattr(ax, "images", ()) or ()) + list(
                getattr(ax, "collections", ()) or ()
            ):
                out = _yield(getattr(artist, "colorbar", None))
                if out is not None:
                    yield out
    except Exception:
        pass


def _refresh_strain_figure_colorbars(fig: Any) -> None:
    """
    Refresh colorbars after clim changes.

    py4DSTEM ``get_strain`` figures often use a horizontal θ colorbar whose tick
    labels pile up. For θ-like panels we replace tickers
    with a small fixed number of evenly spaced ticks and always render labels in
    **degrees** with a ``°`` symbol — matching the ``theta (deg)`` GUI label and
    the ``state.py`` defaults.

    If the underlying mappable stores θ in radians (e.g. py4DSTEM internally
    applied ``deg2rad`` to ``vrange_theta``) tick *positions* remain in radian
    space but tick *labels* are converted to degrees via ``FuncFormatter`` so the
    numeric values and the ``°`` unit are consistent.

    For ε panels stored as **fraction** while the GUI ``vrange`` is in **%**,
    clims are set in fraction space; strain colorbars that advertise ``%`` get
    tick labels multiplied by 100 so they match the GUI (e.g. -5 … 5).
    """
    try:
        import matplotlib.ticker as mticker

        for cb in _iter_figure_matplotlib_colorbars(fig):
            is_theta = _strain_colorbar_associates_theta(cb)
            # Do not call update_normal on θ colorbars: it re-applies py4DSTEM's default
            # locators (e.g. ±2.5° as ±0.0436 rad ticks) and can fight our GUI clim/ticks.
            if not is_theta:
                try:
                    cb.update_normal(cb.mappable)
                except Exception:
                    try:
                        cb.draw_all()
                    except Exception:
                        pass

            if is_theta:
                try:
                    vmin, vmax = cb.mappable.get_clim()
                except Exception:
                    continue
                if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
                    continue

                data_is_radians = False
                try:
                    arr_full = np.asarray(cb.mappable.get_array(), dtype=float).ravel()
                    arr_full = arr_full[np.isfinite(arr_full)]
                    if arr_full.size:
                        data_is_radians = (
                            float(np.nanmax(np.abs(arr_full))) <= float(np.pi) + 0.08
                        )
                    else:
                        data_is_radians = (
                            max(abs(vmin), abs(vmax)) <= float(np.pi) + 0.08
                        )
                except Exception:
                    data_is_radians = max(abs(vmin), abs(vmax)) <= float(np.pi) + 0.08

                if data_is_radians:
                    fmt = mticker.FuncFormatter(
                        lambda x, _pos: f"{np.rad2deg(float(x)):.1f}\u00b0"
                    )
                else:
                    fmt = mticker.StrMethodFormatter("{x:.1f}\u00b0")

                ori = str(getattr(cb, "orientation", "vertical") or "vertical").lower()
                try:
                    cb.ax.minorticks_off()
                except Exception:
                    pass

                ticks = np.linspace(float(vmin), float(vmax), num=5)
                try:
                    cb.set_ticks(ticks)
                    cb.formatter = fmt
                    cb.update_ticks()
                except Exception:
                    pass
                try:
                    if ori.startswith("h"):
                        cb.ax.set_xticks(ticks)
                        cb.ax.xaxis.set_major_formatter(fmt)
                        cb.ax.tick_params(axis="x", labelsize=9)
                        for lbl in cb.ax.get_xticklabels():
                            lbl.set_ha("center")
                    else:
                        cb.ax.set_yticks(ticks)
                        cb.ax.yaxis.set_major_formatter(fmt)
                        cb.ax.tick_params(axis="y", labelsize=9)
                except Exception:
                    pass
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
                continue

            if _strain_colorbar_labels_fraction_as_percent(cb):
                try:
                    vmin, vmax = cb.mappable.get_clim()
                except Exception:
                    continue
                if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
                    continue
                fmt = mticker.FuncFormatter(lambda x, _pos: f"{float(x) * 100.0:.1f}")
                ori = str(getattr(cb, "orientation", "vertical") or "vertical").lower()
                try:
                    cb.ax.minorticks_off()
                except Exception:
                    pass
                ticks = np.linspace(float(vmin), float(vmax), num=5)
                try:
                    cb.set_ticks(ticks)
                    cb.formatter = fmt
                    cb.update_ticks()
                except Exception:
                    pass
                try:
                    if ori.startswith("h"):
                        cb.ax.set_xticks(ticks)
                        cb.ax.xaxis.set_major_formatter(fmt)
                        cb.ax.tick_params(axis="x", labelsize=9)
                        for lbl in cb.ax.get_xticklabels():
                            lbl.set_ha("center")
                    else:
                        cb.ax.set_yticks(ticks)
                        cb.ax.yaxis.set_major_formatter(fmt)
                        cb.ax.tick_params(axis="y", labelsize=9)
                except Exception:
                    pass
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
    except Exception:
        pass


def _apply_get_strain_vrange_to_figures(
    state: WorkflowState,
    figure_nums: list[int],
    *,
    raw_tensor=None,
    log: Callable[[str], None] | None = None,
) -> None:
    """
    Sync color limits with GUI ``vrange`` / ``vrange_theta`` after ``get_strain``.

    Handles both ``AxesImage`` and ``Collection`` (e.g. QuadMesh) maps, refreshes colorbars,
    and chooses fraction vs percent scaling using the raw strain tensor when available.
    """
    if not figure_nums:
        return
    gs = (state.strain_params or {}).get("get_strain") or {}
    vr = gs.get("vrange", [-2, 2])
    vt = gs.get("vrange_theta", [-1.0, 1.0])
    try:
        vmin_p, vmax_p = float(vr[0]), float(vr[1])
        tmin, tmax = float(vt[0]), float(vt[1])
    except Exception:
        return

    tmin_o, tmax_o = _effective_orientation_clims(tmin, tmax, raw_tensor)

    ref_frac = _reference_strain_scale_exx_fractional(raw_tensor)
    maps_for_theta = _strain_maps_dict_from_raw(raw_tensor)
    th_ref = maps_for_theta.get("theta")

    import matplotlib.pyplot as plt

    n_updated = 0
    for num in figure_nums:
        try:
            fig = plt.figure(num)
        except Exception:
            continue
        for ax in fig.axes:
            theta_ax = _axis_title_suggests_theta(ax) or _heatmap_on_ax_matches_reference_array(
                ax, th_ref
            )
            for mappable, amax_display in _iter_strain_heatmap_mappables(ax):
                clims = _strain_panel_clims(
                    vmin_p,
                    vmax_p,
                    tmin_o,
                    tmax_o,
                    theta_ax=theta_ax,
                    amax_display=amax_display,
                    ref_frac=ref_frac,
                )
                if clims is None:
                    continue
                lo, hi = clims
                try:
                    mappable.set_clim(lo, hi)
                    n_updated += 1
                except Exception:
                    pass
        _refresh_strain_figure_colorbars(fig)
        try:
            fig.canvas.draw_idle()
        except Exception:
            try:
                fig.canvas.draw()
            except Exception:
                pass
    if n_updated:
        _log(
            log,
            f"Applied strain color limits to {n_updated} map layer(s) "
            f"(ε display ≈ {vmin_p}..{vmax_p} %, θ clim ≈ {tmin_o}..{tmax_o} (GUI θ {tmin}..{tmax}); ref|εxx|≈{ref_frac:.4g}).",
        )
    elif figure_nums:
        _log(
            log,
            "Strain vrange: no imshow/collection layers matched — py4DSTEM figure layout may differ.",
        )


def _apply_show_orientation_to_figures(
    state: WorkflowState,
    figure_nums: list[int],
    *,
    log: Callable[[str], None] | None = None,
) -> None:
    """
    Hide the theta (orientation) panel + its colorbar when the user has turned
    ``show_orientation`` off, and stretch the remaining exx/eyy/exy panels (and
    their colorbars) to fill the freed space.

    py4DSTEM's ``show_strain`` always draws all 4 panels — there is no native
    kwarg to omit theta — so this is a post-hoc figure edit, same spirit as
    ``_apply_get_strain_vrange_to_figures`` above. Only supported for the linear
    ``horizontal``/``vertical`` layouts, where panels are evenly spaced along one
    axis; ``square`` (2x2 grid) is left with all 4 panels.
    """
    if not figure_nums:
        return
    show = bool((state.strain_params or {}).get("show_orientation", True))
    if show:
        return
    layout = str((state.strain_params or {}).get("get_strain", {}).get("layout", "horizontal"))
    if layout not in ("horizontal", "vertical"):
        return
    horiz = layout == "horizontal"

    import matplotlib.pyplot as plt

    n_hidden = 0
    for num in figure_nums:
        try:
            fig = plt.figure(num)
        except Exception:
            continue
        cb_ax_by_data_ax: dict[Any, Any] = {}
        colorbar_axes: set[Any] = set()
        for cb in _iter_figure_matplotlib_colorbars(fig):
            colorbar_axes.add(cb.ax)
            parent = getattr(cb.mappable, "axes", None)
            if parent is not None:
                cb_ax_by_data_ax[parent] = cb.ax

        # Exclude colorbar axes: their own gradient QuadMesh/AxesImage can otherwise
        # be mistaken for a data panel by _iter_strain_heatmap_mappables (it has no
        # shape guard on the collections branch).
        data_axes = [
            ax for ax in fig.axes
            if ax not in colorbar_axes and list(_iter_strain_heatmap_mappables(ax))
        ]
        theta_ax = next((ax for ax in data_axes if _axis_title_suggests_theta(ax)), None)
        if theta_ax is None or len(data_axes) < 2:
            continue
        remaining = [ax for ax in data_axes if ax is not theta_ax]
        if not remaining:
            continue

        # Use the nominal ("original") box throughout: aspect='equal' images get an
        # auto-shrunk "active" box (get_position(original=False)) that matplotlib
        # re-derives from the nominal one at draw time, so mixing the two here would
        # double-apply the aspect shrink and throw off the redistribution math.
        all_boxes = [ax.get_position(original=True) for ax in data_axes]
        if horiz:
            lo = min(b.x0 for b in all_boxes)
            hi = max(b.x1 for b in all_boxes)
        else:
            lo = min(b.y0 for b in all_boxes)
            hi = max(b.y1 for b in all_boxes)

        remaining_boxes = [ax.get_position(original=True) for ax in remaining]
        if horiz:
            old_lo = min(b.x0 for b in remaining_boxes)
            old_hi = max(b.x1 for b in remaining_boxes)
        else:
            old_lo = min(b.y0 for b in remaining_boxes)
            old_hi = max(b.y1 for b in remaining_boxes)
        old_span = old_hi - old_lo
        if old_span <= 0:
            continue
        scale = (hi - lo) / old_span

        def _remap(v: float) -> float:
            return lo + (v - old_lo) * scale

        theta_ax.set_visible(False)
        theta_cb_ax = cb_ax_by_data_ax.get(theta_ax)
        if theta_cb_ax is not None:
            theta_cb_ax.set_visible(False)

        for ax in remaining:
            cb_ax = cb_ax_by_data_ax.get(ax)
            for target in (ax, cb_ax):
                if target is None:
                    continue
                tbox = target.get_position(original=True)
                if horiz:
                    new_x0, new_x1 = _remap(tbox.x0), _remap(tbox.x1)
                    target.set_position([new_x0, tbox.y0, new_x1 - new_x0, tbox.height])
                else:
                    new_y0, new_y1 = _remap(tbox.y0), _remap(tbox.y1)
                    target.set_position([tbox.x0, new_y0, tbox.width, new_y1 - new_y0])
        n_hidden += 1
        try:
            fig.canvas.draw_idle()
        except Exception:
            try:
                fig.canvas.draw()
            except Exception:
                pass
    if n_hidden:
        _log(log, f"Hid orientation (theta) panel on {n_hidden} strain figure(s).")


def capture_reference_g12_from_strainmap(
    strainmap: Any,
    roi_mask: np.ndarray,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Median fitted ``(g1, g2)`` over a real-space ROI on an existing ``StrainMap``
    (after ``fit_basis_vectors`` / ``get_strain``). Use on a **reference** scan
    (e.g. naked Si), then pass the returned pair as ``gvects`` when computing
    strain on other scans with the **same** basis / calibration convention.
    """
    if strainmap is None:
        raise RuntimeError("StrainMap is missing: run full strain on the reference scan first.")
    mask = np.asarray(roi_mask, dtype=bool)
    rshape = getattr(strainmap, "rshape", None)
    if rshape is None:
        raise RuntimeError("StrainMap has no rshape.")
    if tuple(mask.shape) != tuple(rshape):
        raise RuntimeError(
            f"ROI mask shape {mask.shape} does not match StrainMap rshape {tuple(rshape)}."
        )
    if int(mask.sum()) < 1:
        raise RuntimeError("ROI mask is empty.")
    if not hasattr(strainmap, "get_reference_g1g2"):
        raise RuntimeError("This py4DSTEM StrainMap has no get_reference_g1g2; update py4DSTEM?")
    g1, g2 = strainmap.get_reference_g1g2(mask)
    a1 = np.asarray(g1, dtype=np.float64).ravel()
    a2 = np.asarray(g2, dtype=np.float64).ravel()
    if a1.size < 2 or a2.size < 2:
        raise RuntimeError(f"Unexpected reference vector shapes: g1={a1!r}, g2={a2!r}.")
    a1 = a1[:2].copy()
    a2 = a2[:2].copy()
    _log(log, f"Captured external strain reference g1={a1.tolist()}, g2={a2.tolist()}")
    return a1, a2


def _braggvectors_have_calibrated_origin(braggpeaks: Any) -> bool:
    """True if py4DSTEM ``StrainMap`` will accept these ``braggvectors`` (origin calibration set)."""
    try:
        cal = getattr(braggpeaks, "calibration", None)
        if cal is None:
            return False
        origin = getattr(cal, "origin", None)
        if origin is not None:
            return True
        if hasattr(cal, "get_origin"):
            try:
                return cal.get_origin() is not None
            except Exception:
                return False
    except Exception:
        return False
    return False


def _bragg_calstate_flag(braggpeaks: Any, *names: str) -> bool | None:
    """Read a boolean from ``braggpeaks.calstate`` (object or dict). None if absent."""
    if braggpeaks is None:
        return None
    cs = getattr(braggpeaks, "calstate", None)
    if cs is None:
        return None
    if isinstance(cs, dict):
        for n in names:
            if n in cs:
                return bool(cs[n])
        return None
    for n in names:
        if hasattr(cs, n):
            return bool(getattr(cs, n))
    return None


def _write_bragg_calstate_flags(braggpeaks: Any, **flags: bool) -> None:
    """Set one or more ``calstate`` booleans and refresh calibration (``setcal``)."""
    if braggpeaks is None or not flags:
        return
    cs = getattr(braggpeaks, "calstate", None)
    if cs is None:
        return
    key_aliases = {
        "ellipse": ("ellipse",),
        "origin": ("origin", "center"),
        "qpixel": ("qpixel", "Q_pixel", "q_pixel", "pixel"),
        "basis": ("basis", "rotate"),
    }
    if isinstance(cs, dict):
        for key, val in flags.items():
            for name in key_aliases.get(key, (key,)):
                cs[name] = bool(val)
    else:
        for key, val in flags.items():
            wrote = False
            for name in key_aliases.get(key, (key,)):
                if hasattr(cs, name):
                    setattr(cs, name, bool(val))
                    wrote = True
            if not wrote and hasattr(cs, key):
                setattr(cs, key, bool(val))
    try:
        braggpeaks.setcal()
    except Exception:
        pass


# Prior calibration steps applied by single-scan «Apply all prev cals» (notebook order).
_PREV_CAL_STEPS: tuple[str, ...] = ("origin", "ellipse", "qpixel", "basis")

_PREV_CAL_UP_TO: dict[str, tuple[str, ...]] = {
    "ellipse": ("origin",),
    "qpixel": ("origin", "ellipse"),
    "basis": ("origin", "ellipse", "qpixel"),
    "strain": ("origin", "ellipse", "qpixel", "basis"),
}


def mark_ellipse_calibration_skipped(
    state: WorkflowState,
    log: Callable[[str], None] | None = None,
) -> None:
    """Record optional ellipse as unused (``calstate.ellipse=False``) and continue."""
    bp = require_braggpeaks(state)
    _write_bragg_calstate_flags(bp, ellipse=False)
    _log(log, f"Ellipse calibration skipped (optional). calstate={getattr(bp, 'calstate', None)}")


def apply_all_prev_cals_step(
    state: WorkflowState,
    *,
    up_to: str,
    origin_sampling: int | None = None,
    q_pixel_size: float | None = None,
    q_pixel_units: str = "A^-1",
    strain_basis_params: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Apply all calibration steps before ``up_to`` in notebook order:
    Origin → ellipse (optional) → Q pixel → basis.

    Ellipse: if ``state.p_ellipse`` is set, apply it; otherwise set ``calstate.ellipse=False``
    and continue. Strain mapping is never run here (only prior calibrations).
    """
    key = (up_to or "").strip().lower()
    if key not in _PREV_CAL_UP_TO:
        raise ValueError(f"up_to must be one of {sorted(_PREV_CAL_UP_TO)}; got {up_to!r}")

    steps = _PREV_CAL_UP_TO[key]
    applied: list[str] = []
    skipped: list[str] = []
    bp = require_braggpeaks(state)

    if "origin" in steps:
        if state.center_guess is None:
            raise RuntimeError("Set origin center_guess (y,x) before applying prior calibrations.")
        if not _braggvectors_have_calibrated_origin(bp):
            samp = int(origin_sampling if origin_sampling is not None else state.origin_sampling)
            run_origin_correction_step(state, sampling=samp, log=log)
            applied.append("origin")
        else:
            skipped.append("origin (already calibrated)")
            _log(log, "Origin: already calibrated on braggpeaks — skipped.")

    if "ellipse" in steps:
        if state.p_ellipse is not None:
            apply_ellipse_step(state, log=log)
            applied.append("ellipse")
        else:
            mark_ellipse_calibration_skipped(state, log=log)
            skipped.append("ellipse (optional, not fitted)")

    if "qpixel" in steps:
        px = q_pixel_size
        if px is None:
            px = getattr(state, "q_pixel_size", None)
        try:
            px_f = float(px)
        except (TypeError, ValueError):
            px_f = float("nan")
        if not (px_f == px_f and px_f > 0):
            raise RuntimeError(
                "Set a positive Q pixel size (Step 8 sliders or params file) before applying prior calibrations."
            )
        set_q_pixel_size_step(state, px_f, units=str(q_pixel_units or state.q_pixel_units), log=log)
        applied.append("qpixel")

    if "basis" in steps:
        if strain_basis_params is None:
            sbp = getattr(state, "strain_basis_params", None)
            if not isinstance(sbp, dict) or not sbp.get("choose_basis_vectors"):
                raise RuntimeError("Strain basis parameters are not set.")
            strain_basis_params = sbp
        cbv = (strain_basis_params or {}).get("choose_basis_vectors") or {}
        vp = (cbv.get("vis_params") or {}) if isinstance(cbv, dict) else {}
        update_strain_basis_params(
            state,
            min_spacing=int(cbv.get("minSpacing", 5)),
            min_absolute_intensity=int(cbv.get("minAbsoluteIntensity", 80)),
            max_num_peaks=int(cbv.get("maxNumPeaks", 60)),
            edge_boundary=int(cbv.get("edgeBoundary", 4)),
            vmin=float(vp.get("vmin", 0.0)),
            vmax=float(vp.get("vmax", 0.995)),
            qr_rotation=float(strain_basis_params.get("qr_rotation", 0.0)),
            qr_flip=bool(strain_basis_params.get("qr_flip", False)),
            manual_enabled=bool(strain_basis_params.get("manual_enabled", False)),
            index_origin=int(cbv.get("index_origin", 0)),
            index_g1=int(cbv.get("index_g1", 0)),
            index_g2=int(cbv.get("index_g2", 0)),
            log=log,
        )
        applied.append("basis")

    bp = require_braggpeaks(state)
    _log(
        log,
        f"Apply all prev cals (up_to={key}): applied={applied or ['—']}, "
        f"skipped={skipped or ['—']}; calstate={getattr(bp, 'calstate', None)}",
    )
    return {
        "up_to": key,
        "applied": applied,
        "skipped": skipped,
        "calstate": getattr(bp, "calstate", None),
    }


def _bragg_has_q_pixel_calibration(braggpeaks: Any, state: WorkflowState) -> bool:
    flag = _bragg_calstate_flag(braggpeaks, "qpixel", "Q_pixel", "q_pixel")
    if flag is True:
        return True
    cal = getattr(braggpeaks, "calibration", None) if braggpeaks is not None else None
    if cal is not None:
        for meth in ("get_Q_pixel_size", "get_q_pixel_size"):
            if hasattr(cal, meth):
                try:
                    v = getattr(cal, meth)()
                    if v is not None and float(v) > 0:
                        return True
                except Exception:
                    pass
    try:
        qps = float(getattr(state, "q_pixel_size", 0) or 0)
        if qps > 0 and flag is not False:
            return True
    except Exception:
        pass
    return False


def _bragg_has_basis_calibration(braggpeaks: Any, state: WorkflowState) -> bool:
    flag = _bragg_calstate_flag(braggpeaks, "basis")
    if flag is True:
        return True
    sbp = getattr(state, "strain_basis_params", None) or {}
    if isinstance(sbp, dict) and sbp:
        if sbp.get("g1_qxy") is not None and sbp.get("g2_qxy") is not None:
            return True
    return False


def single_scan_cal_ui_flags(state: WorkflowState) -> dict[str, str]:
    """
    UI strip states for single-scan bottom bar.

    Keys: origin, ellipse, qpx, basis, strain, stress, lines.
    Values: ``applied`` | ``staged`` | ``pending`` | ``unused``.
    """
    bp = getattr(state, "braggpeaks", None)
    out: dict[str, str] = {}

    if bp is None:
        for k in ("origin", "ellipse", "qpx", "basis", "strain", "stress", "lines"):
            out[k] = "unused"
        return out

    if _braggvectors_have_calibrated_origin(bp):
        out["origin"] = "applied"
    elif state.center_guess is not None:
        out["origin"] = "staged"
    else:
        out["origin"] = "pending"

    ell_cs = _bragg_calstate_flag(bp, "ellipse")
    if ell_cs is False:
        out["ellipse"] = "unused"
    elif ell_cs is True:
        out["ellipse"] = "applied"
    elif getattr(state, "p_ellipse", None) is not None:
        out["ellipse"] = "staged"
    else:
        out["ellipse"] = "pending"

    if _bragg_has_q_pixel_calibration(bp, state):
        out["qpx"] = "applied"
    else:
        try:
            if float(state.q_px.get()) > 0 if hasattr(state, "q_px") else False:
                out["qpx"] = "staged"
            else:
                out["qpx"] = "pending"
        except Exception:
            out["qpx"] = "pending"

    if _bragg_has_basis_calibration(bp, state):
        out["basis"] = "applied"
    elif isinstance(getattr(state, "strain_basis_params", None), dict) and state.strain_basis_params:
        out["basis"] = "staged"
    else:
        out["basis"] = "pending"

    raw = getattr(state, "strain_raw", None) or {}
    if isinstance(raw, dict) and raw.get("without_roi") is not None:
        out["strain"] = "applied"
    elif (state.strain_params or {}).get("get_strain"):
        out["strain"] = "staged"
    else:
        out["strain"] = "pending"

    st = getattr(state, "stress_tensors_pa", None) or {}
    if isinstance(st, dict) and st:
        out["stress"] = "applied"
    else:
        out["stress"] = "pending"

    lp = getattr(state, "line_profiles_px", None) or {}
    if lp:
        out["lines"] = "applied"
    else:
        out["lines"] = "pending"

    return out


def _ensure_braggpeaks_origin_for_strain(
    state: WorkflowState,
    log: Callable[[str], None] | None = None,
) -> None:
    """
    py4DSTEM ``StrainMap`` asserts ``braggvectors.calibration.origin is not None``.
    Params import may set ``center_guess`` without running ``measure_origin`` / ``fit_origin``.
    When origin is missing but ``center_guess`` exists, run :func:`run_origin_correction_step`.
    """
    bp = require_braggpeaks(state)
    if _braggvectors_have_calibrated_origin(bp):
        return
    if state.center_guess is None:
        raise RuntimeError(
            "braggpeaks has no calibrated origin (required before StrainMap). "
            "Run Step 9 origin correction on this dataset, or load calibration params that include "
            "origin center_guess and enable applying them so center_guess is set, then retry strain."
        )
    _log(
        log,
        "braggpeaks lacks a calibrated origin; running origin correction using center_guess "
        f"{state.center_guess} (sampling={state.origin_sampling}).",
    )
    run_origin_correction_step(state, sampling=state.origin_sampling, log=log)


def _prepare_gvects_for_get_strain(strainmap: Any, gvects: Any) -> Any:
    """
    ``gvects`` for ``StrainMap.get_strain``:
    - ``tuple`` of two length-2 vectors → passed through as explicit ``(g1, g2)`` reference
    - otherwise → boolean ROI mask sanitized against finite ``g1g2_map`` pixels
    """
    if gvects is None:
        return None
    if isinstance(gvects, tuple) and len(gvects) == 2:
        g1, g2 = gvects
        a1 = np.asarray(g1, dtype=np.float64).ravel()
        a2 = np.asarray(g2, dtype=np.float64).ravel()
        if a1.size < 2 or a2.size < 2:
            raise RuntimeError("External g1,g2 must each have at least two components (qx, qy).")
        return (a1[:2].copy(), a2[:2].copy())
    return _sanitize_gvects(strainmap, gvects)


def compute_strain_map_step(
    state: WorkflowState,
    use_roi: bool = False,
    gvects=None,
    label_override: str | None = None,
    mask_braggpeaks_outside_scan_roi: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compute strain mapping and capture a figure + a primary 2D array.

    IMPORTANT: In Tk apps on Windows, plotting via pyplot/TkAgg inside py4DSTEM
    can raise "main thread is not in main loop". We therefore try to run
    get_strain() in a non-plotting mode first and build a lightweight figure
    ourselves. If the installed py4DSTEM does not support plot suppression,
    we fall back to the original plotting behavior.
    """

    import py4DSTEM

    # IMPORTANT notebook behavior:
    # - Optional scan-sector mask (old notebooks): drop patterns outside a real-space ROI via mask_in_R.
    # - Step 5 ellipse/histogram ROI still applies only when use_roi and gvects is None (see below).
    # - ROI recompute (gvects set) does not apply Step 5 mask to braggpeaks unless you also enable scan-sector mask.
    braggpeaks = require_braggpeaks(state)
    _ensure_braggpeaks_origin_for_strain(state, log=log)
    if mask_braggpeaks_outside_scan_roi:
        sm = state.strain_scan_roi_mask
        if sm is None:
            raise RuntimeError(
                "mask_braggpeaks_outside_scan_roi is True but strain_scan_roi_mask is not set. "
                "Pick a sector on ADF/BF in Step 13 (or copy Step 5 ROI)."
            )
        braggpeaks = _mask_braggpeaks_keep_scan_sector(braggpeaks, sm, log=log)
    braggpeaks = _braggpeaks_for_roi_from_bp(braggpeaks, state, use_roi=(use_roi and gvects is None))
    label = str(label_override) if label_override else ("with_roi" if use_roi else "without_roi")
    strainmap = py4DSTEM.StrainMap(braggvectors=braggpeaks)

    # py4DSTEM's strain methods call plt.show() internally in some versions
    # (e.g. set_max_peak_spacing). In a Tk app on Windows this can trigger
    # TkAgg window creation and crash with "main thread is not in main loop".
    # We therefore suppress pyplot show/pause during these calls and render
    # our own figure afterwards.
    _with_plot_suppressed(lambda: _choose_basis_vectors_safely(strainmap, state.strain_basis_params["choose_basis_vectors"]))
    _with_plot_suppressed(lambda: strainmap.set_max_peak_spacing(**state.strain_params["set_max_peak_spacing"]))
    _with_plot_suppressed(lambda: strainmap.fit_basis_vectors(**state.strain_params["fit_basis_vectors"]))
    def _run_get_strain():
        kwargs = dict(state.strain_params["get_strain"])
        if gvects is not None:
            kwargs["gvects"] = _prepare_gvects_for_get_strain(strainmap, gvects)
        strainmap.get_strain(**kwargs)

    # Capture the figure(s) produced by py4DSTEM (notebook style),
    # while suppressing plt.show() so it doesn't pop up a Tk window.
    before = set(plt.get_fignums())
    _with_plot_suppressed(_run_get_strain)
    after = set(plt.get_fignums())
    new_nums = sorted(after - before)
    # Second and later runs often redraw into the same figure IDs; then new_nums is empty
    # and we must still refresh those figures (and apply vrange).
    if new_nums:
        target_nums: list[int] = new_nums
    elif before & after:
        target_nums = sorted(before & after)
    else:
        target_nums = sorted(after)
    figures = [plt.figure(n) for n in target_nums]
    fig = figures[-1] if figures else None
    raw = _extract_strainmap_g1g2_data(strainmap)
    raw = _sanitize_strain_raw(raw, state, label, log=log)
    _sync_pyplot_strain_figures_from_raw(target_nums, raw)
    _apply_get_strain_vrange_to_figures(state, target_nums, raw_tensor=raw, log=log)
    _apply_show_orientation_to_figures(state, target_nums, log=log)
    arr = _extract_primary_strain_array(strainmap, raw=raw)

    if use_roi:
        state.strainmap_roi = strainmap
    else:
        state.strainmap_full = strainmap
    if fig is not None:
        state.strain_figures[label] = fig
    state.strain_arrays[label] = arr
    state.strain_raw[label] = raw
    _log(log, f"Computed strain mapping {label}.")
    return {"label": label, "strainmap": strainmap, "figure": fig, "figures": figures, "array": arr, "raw": raw}


def strain_tensor_components_from_state(
    state: WorkflowState,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unitless (ε_xx, ε_yy, ε_xy) from ``state.strain_raw[label]``.
    Stack channel order: 0 = ε_yy, 1 = ε_xx, 2 = ε_xy (py4DSTEM / notebook).
    """
    raw = state.strain_raw.get(label)
    if raw is None:
        raise RuntimeError(
            f"No strain tensor at strain_raw[{label!r}]. Compute that strain map first."
        )
    S = np.squeeze(np.asarray(raw))
    if S.ndim != 3:
        raise RuntimeError(
            "Stress analysis needs the full in-plane strain stack (3 components). "
            f"Got shape {S.shape} for label {label!r}."
        )
    eyy = _strain_channel(S, 0)
    exx = _strain_channel(S, 1)
    exy = _strain_channel(S, 2)
    return (
        np.asarray(exx, dtype=np.float64),
        np.asarray(eyy, dtype=np.float64),
        np.asarray(exy, dtype=np.float64),
    )


def compute_stress_analysis_step(
    state: WorkflowState,
    label: str,
    *,
    mode: str,
    c11_pa: float,
    c12_pa: float,
    c44_pa: float,
    overlay_strain: bool = False,
    vmin_gpa: float | None = None,
    vmax_gpa: float | None = None,
    units: str = "GPa",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """σ from ε via cubic Hooke's law (plane stress or plane strain). θ not used.

    ``units`` ("GPa"|"MPa") sets the stress-map display units; ``vmin_gpa``/
    ``vmax_gpa`` are the symmetric colour range IN THOSE display units (None = auto).
    """

    try:
        from .stress_analysis import build_stress_maps_figure, compute_stress
    except ImportError:
        from stress_analysis import build_stress_maps_figure, compute_stress

    exx, eyy, exy = strain_tensor_components_from_state(state, label)
    sigma = compute_stress(exx, eyy, exy, c11_pa, c12_pa, c44_pa, mode=mode)
    state.stress_tensors_pa[label] = {k: np.asarray(v, dtype=np.float64) for k, v in sigma.items()}
    state.stress_meta[label] = {
        "mode": str(mode),
        "C11_Pa": float(c11_pa),
        "C12_Pa": float(c12_pa),
        "C44_Pa": float(c44_pa),
    }
    gs = (state.strain_params or {}).get("get_strain", {})
    vr = gs.get("vrange", [-5.0, 5.0])
    try:
        strain_vmm = (float(vr[0]), float(vr[1]))
    except Exception:
        strain_vmm = (-5.0, 5.0)
    overlay_tuple = (eyy, exx, exy) if overlay_strain else None
    mode_label = str(mode).replace("_", " ")

    # Gather line segments to overlay on the stress maps.
    # Priority: fixed_line_profiles_px (global batch segments) > line_profiles_px (per-scan).
    _segs: dict = {}
    try:
        fp = getattr(state, "fixed_line_profiles_px", None)
        lp = getattr(state, "line_profiles_px", None)
        _raw_segs = fp if (isinstance(fp, dict) and fp) else (lp or {})
        for k, v in _raw_segs.items():
            try:
                # Normalise: value is ((x0,y0),(x1,y1)) or {lid: seg} etc.
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    _segs[k] = (tuple(v[0]), tuple(v[1]))
            except Exception:
                pass
    except Exception:
        pass
    _lp_width = 3
    try:
        _lp_width = int(getattr(state, "line_profile_width", 3) or 3)
    except Exception:
        pass

    fig = build_stress_maps_figure(
        sigma,
        mode_label=f"{mode_label} | {label}",
        vmin_gpa=vmin_gpa,
        vmax_gpa=vmax_gpa,
        units=units,
        overlay_strain_percent=overlay_tuple,
        strain_vminmax_percent=strain_vmm,
        line_segments=_segs or None,
        line_profile_width=_lp_width,
    )
    state.stress_figures[label] = fig
    smax = float(np.nanmax(np.abs(sigma["sigma_xx"]))) / 1e9
    _log(log, f"Stress ({label}): {mode_label}, max|σxx|≈{smax:.4g} GPa.")
    return {
        "label": label,
        "sigma_pa": sigma,
        "figure": fig,
        "figures": [fig],
    }


def _sanitize_gvects(strainmap, gvects):
    """
    Make ROI mask safer for py4DSTEM's reference fit:
    - ensure boolean array
    - drop pixels where g1g2_map is non-finite (common cause of SVD failure)
    - optional erosion if scipy is available (reduces edge artifacts)
    """
    mask = np.asarray(gvects).astype(bool)
    try:
        gmap = getattr(strainmap, "g1g2_map", None)
        gdata = getattr(gmap, "data", gmap)
        garr = np.asarray(gdata, dtype=float)
        if garr.ndim >= 3:
            finite = np.isfinite(garr).all(axis=tuple(range(garr.ndim - 2)))
            mask = mask & finite
    except Exception:
        pass
    try:
        from scipy.ndimage import binary_erosion

        mask = binary_erosion(mask, iterations=1)
    except Exception:
        pass
    if int(mask.sum()) < 25:
        raise RuntimeError(f"ROI too small/invalid after filtering (valid pixels={int(mask.sum())}). Pick a larger ROI.")
    return mask


def _extract_strainmap_g1g2_data(strainmap):
    for name in ("strainmap_g1g2", "strainmap", "strain_map"):
        if hasattr(strainmap, name):
            obj = getattr(strainmap, name)
            return getattr(obj, "data", obj)
    return None


_STRAIN_SENTINEL_ABS_MAX = 1e12
# Vacuum / failed-fit artefacts often land at exactly ±100% (|ε|=1); real maps rarely exceed this.
_STRAIN_FRACTION_UNPHYSICAL_ABS = 1.0


def _is_tensor_strain_field_name(name: str) -> bool:
    n = str(name).lower()
    if any(x in n for x in ("theta", "orient", "rotation", "angle", "phase")):
        return False
    return any(x in n for x in ("exx", "eyy", "exy", "e_xx", "e_yy", "e_xy", "eps_xx", "eps_yy", "eps_xy"))


def _scan_void_mask_from_virtual_image(state: WorkflowState, shape_hw: tuple[int, int]) -> np.ndarray | None:
    """ADF low-intensity pixels (same shape as strain) approximate vacuum / no sample."""
    H, W = int(shape_hw[0]), int(shape_hw[1])
    vis = getattr(state, "virtual_images", None)
    if not vis:
        return None
    # ADF: vacuum is dark. Avoid BF (often inverted) and DP mean (can be misleading).
    for key in ("annular_dark_field",):
        obj = vis.get(key)
        if obj is None:
            continue
        try:
            arr = np.asarray(getattr(obj, "data", obj), dtype=float)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        if arr.shape[0] != H or arr.shape[1] != W:
            continue
        flat = np.nan_to_num(arr.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        if flat.size == 0 or float(np.nanmax(flat)) <= 0:
            continue
        lo = float(np.percentile(flat, 15.0))
        hi = float(np.percentile(flat, 99.5))
        scale = max(hi - lo, hi * 1e-6, 1e-12)
        n = np.clip((arr - lo) / scale, 0.0, 1.0)
        dark = np.asarray(n < 0.06, dtype=bool)
        return dark | ~np.isfinite(arr)
    return None


def _sanitize_strain_raw(
    raw: Any,
    state: WorkflowState,
    label: str,
    log: Callable[[str], None] | None = None,
) -> Any:
    """
    In-place cleanup: NaN/Inf and absurd sentinels → 0.

    Strain ROI is only for reference fitting (``gvects``); do not blank exterior pixels here.

    Also zeros tensor strain ε where |ε|≥1 (100% artefacts from vacío/div0) or on low-intensity
    scan pixels (ADF/BF void mask when shapes match virtual images).
    """
    if raw is None:
        return None

    arr = np.asarray(raw)
    if arr.dtype.names:
        names_lo = {str(n).lower(): n for n in arr.dtype.names}
        refs: list[np.ndarray] = []
        for cand in ("eyy", "e_yy", "eps_yy", "exx", "e_xx", "eps_xx", "exy", "e_xy", "eps_xy"):
            k = names_lo.get(cand)
            if k is None:
                continue
            sl = np.asarray(arr[k], dtype=np.float64)
            if sl.ndim == 2:
                refs.append(sl)
        if not refs:
            return raw
        H, W = refs[0].shape
        invalid = np.zeros((H, W), dtype=bool)
        void = _scan_void_mask_from_virtual_image(state, (H, W))
        for key in arr.dtype.names:
            sl = np.asarray(arr[key], dtype=np.float64)
            if sl.ndim != 2 or sl.shape != (H, W):
                continue
            invalid |= ~np.isfinite(sl)
            invalid |= np.abs(sl) > _STRAIN_SENTINEL_ABS_MAX

        mask_base = invalid if void is None else (invalid | void)
        for key in arr.dtype.names:
            sl = np.asarray(arr[key], dtype=np.float64)
            if sl.ndim != 2 or sl.shape != (H, W):
                continue
            sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
            bad = mask_base.copy()
            if _is_tensor_strain_field_name(str(key)):
                bad |= np.abs(sl) >= _STRAIN_FRACTION_UNPHYSICAL_ABS
            sl[bad] = 0.0
            try:
                arr[key][:] = sl
            except Exception:
                try:
                    setattr(arr, key, sl)
                except Exception:
                    pass
        _log(log, f"Sanitized structured strain tensor ({label}): invalid / vacuum / unphysical ε → 0.")
        return raw

    work = np.asarray(raw, dtype=np.float64)
    if work.ndim != 3:
        return raw

    ch_first = work.shape[0] <= 8 and work.shape[-1] > 8
    ch_last = work.shape[-1] <= 8 and not ch_first

    if ch_last:
        if work.shape[-1] < 3:
            return raw
        H, W = int(work.shape[0]), int(work.shape[1])
        void = _scan_void_mask_from_virtual_image(state, (H, W))
        S3 = work[..., :3]
        invalid = ~np.isfinite(S3).all(axis=-1)
        invalid |= np.any(np.abs(S3) > _STRAIN_SENTINEL_ABS_MAX, axis=-1)
        invalid |= np.any(np.abs(S3) >= _STRAIN_FRACTION_UNPHYSICAL_ABS, axis=-1)
        if void is not None:
            invalid |= void
        work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
        work[invalid] = 0.0
        try:
            np.copyto(np.asarray(raw, dtype=np.float64), work)
        except Exception:
            np.copyto(np.asarray(raw), work)
        _log(log, f"Sanitized strain stack ({label}, channels-last): invalid → 0.")
        return raw

    if ch_first and work.shape[0] >= 3:
        H, W = int(work.shape[1]), int(work.shape[2])
        void = _scan_void_mask_from_virtual_image(state, (H, W))
        S3 = work[:3, ...]
        invalid = ~np.isfinite(S3).all(axis=0)
        invalid |= np.any(np.abs(S3) > _STRAIN_SENTINEL_ABS_MAX, axis=0)
        invalid |= np.any(np.abs(S3) >= _STRAIN_FRACTION_UNPHYSICAL_ABS, axis=0)
        if void is not None:
            invalid |= void
        work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
        work[:, invalid] = 0.0
        try:
            np.copyto(np.asarray(raw, dtype=np.float64), work)
        except Exception:
            np.copyto(np.asarray(raw), work)
        _log(log, f"Sanitized strain stack ({label}, channels-first): invalid → 0.")
        return raw

    return raw


def _sanitize_strain_values(arr: np.ndarray) -> np.ndarray:
    """Non-finite and absurd magnitudes → NaN for line-profile nanmean."""
    a = np.asarray(arr, dtype=np.float64)
    out = a.copy()
    bad = ~np.isfinite(out) | (np.abs(out) > _STRAIN_SENTINEL_ABS_MAX)
    bad |= np.abs(out) >= _STRAIN_FRACTION_UNPHYSICAL_ABS
    out[bad] = np.nan
    return out


def _strain_maps_dict_from_raw(raw: Any) -> dict[str, np.ndarray]:
    """Maps keys eyy, exx, exy, theta → 2D arrays for figure sync."""
    out: dict[str, np.ndarray] = {}
    if raw is None:
        return out
    arr = np.asarray(raw)
    if arr.dtype.names:
        names_lo = {str(n).lower(): n for n in arr.dtype.names}
        for alias, keys in (
            ("eyy", ("eyy", "e_yy", "eps_yy")),
            ("exx", ("exx", "e_xx", "eps_xx")),
            ("exy", ("exy", "e_xy", "eps_xy")),
            ("theta", ("theta", "rotation", "angle", "orient")),
        ):
            for k in keys:
                nk = names_lo.get(k)
                if nk is None:
                    continue
                sl = np.asarray(arr[nk], dtype=np.float64)
                if sl.ndim == 2:
                    out[alias] = sl
                    break
        return out

    work = np.asarray(raw, dtype=np.float64)
    if work.ndim != 3:
        return out
    if work.shape[-1] >= 3 and work.shape[-1] <= 8:
        out["eyy"] = np.asarray(work[..., 0], dtype=np.float64)
        out["exx"] = np.asarray(work[..., 1], dtype=np.float64)
        out["exy"] = np.asarray(work[..., 2], dtype=np.float64)
        if work.shape[-1] >= 4:
            out["theta"] = np.asarray(work[..., 3], dtype=np.float64)
    elif work.shape[0] >= 3 and work.shape[0] <= 8:
        out["eyy"] = np.asarray(work[0, ...], dtype=np.float64)
        out["exx"] = np.asarray(work[1, ...], dtype=np.float64)
        out["exy"] = np.asarray(work[2, ...], dtype=np.float64)
        if work.shape[0] >= 4:
            out["theta"] = np.asarray(work[3, ...], dtype=np.float64)
    return out


def _panel_key_from_axes_title(title: str | None) -> str | None:
    if not title:
        return None
    t = title.lower()
    full = title
    if "theta" in t or "orient" in t or "θ" in full:
        return "theta"
    if "xy" in t and "xx" not in t and "yy" not in t:
        return "exy"
    if "xx" in t:
        return "exx"
    if "yy" in t:
        return "eyy"
    return None


def _sync_pyplot_strain_figures_from_raw(figure_nums: list[int], raw: Any) -> None:
    maps = _strain_maps_dict_from_raw(raw)
    if not maps:
        return
    import matplotlib.pyplot as plt

    for num in figure_nums:
        try:
            fig = plt.figure(num)
        except Exception:
            continue
        for ax in fig.axes:
            key = _panel_key_from_axes_title(ax.get_title())
            if key is None or key not in maps:
                continue
            z = maps[key]
            if z.ndim != 2:
                continue
            for im in getattr(ax, "images", ()) or ():
                try:
                    cur = im.get_array()
                    if cur is not None and np.asarray(cur).shape == z.shape:
                        im.set_data(z)
                except Exception:
                    pass
            for coll in getattr(ax, "collections", ()) or ():
                try:
                    ca = coll.get_array()
                    if ca is None:
                        continue
                    ca = np.asarray(ca)
                    if ca.size == z.size:
                        coll.set_array(np.asarray(z, dtype=float).ravel())
                except Exception:
                    pass


def _strain_channel(data, channel: int) -> np.ndarray:
    arr = np.asarray(data)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D strain data after squeeze, got shape {arr.shape}")
    if arr.shape[-1] > channel and arr.shape[-1] <= 8:
        return np.asarray(arr[..., channel], dtype=float)
    if arr.shape[0] > channel and arr.shape[0] <= 8:
        return np.asarray(arr[channel, ...], dtype=float)
    raise ValueError(f"Cannot identify channel axis in strain data shape {arr.shape}")


def _extract_primary_strain_array(strainmap, raw=None) -> np.ndarray:
    # Notebook-compatible: strainmap.strainmap_g1g2.data
    data_sources = []
    if raw is not None:
        data_sources.append(raw)
    for name in ("strainmap_g1g2", "strainmap", "strain_map"):
        if hasattr(strainmap, name):
            data_sources.append(getattr(getattr(strainmap, name), "data", getattr(strainmap, name)))

    for data in data_sources:
        if data is None:
            continue
        try:
            arr = np.asarray(data)
            # If this is the notebook-style 3-channel strain tensor, prefer εxx (channel=1)
            if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
                try:
                    return _strain_channel(arr, 1)
                except Exception:
                    return np.asarray(arr[0], dtype=float)
                # Direct 2D float image
                if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                    return np.asarray(arr, dtype=float)
                # 3D stack: pick xx-like first plane
                if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
                    return np.asarray(arr[0], dtype=float)
                # Structured array (common for strainmap_g1g2.data)
                if arr.dtype.names:
                    # Prefer typical fields in order
                    for field in (
                        "exx",
                        "e_xx",
                        "eps_xx",
                        "eps11",
                        "e11",
                        "eyy",
                        "e_yy",
                        "eps_yy",
                        "exy",
                        "e_xy",
                        "eps_xy",
                        "theta",
                    ):
                        if field in arr.dtype.names:
                            vals = np.asarray(arr[field], dtype=float)
                            if vals.ndim == 2:
                                return vals
                            # 1D flattened map → reshape using rshape if available
                            rshape = getattr(strainmap, "rshape", None) or getattr(strainmap, "Rshape", None)
                            if vals.ndim == 1 and rshape is not None and int(np.prod(rshape)) == vals.size:
                                return vals.reshape(tuple(int(v) for v in rshape))
        except Exception:
            pass
    """
    Best-effort extraction of a representative 2D strain array from a StrainMap,
    across py4DSTEM versions.
    """

    # Common attribute names seen in practice
    for name in (
        "strain",
        "eps_xx",
        "eps_yy",
        "eps_xy",
        "e_xx",
        "e_yy",
        "e_xy",
        "exx",
        "eyy",
        "exy",
    ):
        if hasattr(strainmap, name):
            try:
                arr = np.asarray(getattr(strainmap, name), dtype=float)
                if arr.ndim == 2:
                    return arr
                if arr.ndim == 3 and arr.shape[0] in (3, 4):
                    # pick xx
                    return np.asarray(arr[0], dtype=float)
            except Exception:
                pass
    # Sometimes stored in a dict-like
    for name in ("strain_maps", "strain_map", "maps", "results"):
        if hasattr(strainmap, name):
            try:
                obj = getattr(strainmap, name)
                if isinstance(obj, dict):
                    for key in ("eps_xx", "exx", "e_xx", "xx"):
                        if key in obj:
                            arr = np.asarray(obj[key], dtype=float)
                            if arr.ndim == 2:
                                return arr
            except Exception:
                pass
    raise RuntimeError("Could not extract a 2D strain array from StrainMap after get_strain(plot=False).")


def _with_agg_backend(fn):
    """
    Temporarily switch pyplot backend to Agg for the duration of `fn`.
    This prevents TkAgg from creating/destroying Tk windows inside py4DSTEM.
    """
    try:
        import matplotlib.pyplot as _plt

        prev = None
        try:
            prev = _plt.get_backend()
        except Exception:
            prev = None
        try:
            _plt.switch_backend("Agg")
        except Exception:
            # If switch_backend fails, still run fn; it may still work.
            return fn()
        try:
            return fn()
        finally:
            try:
                if prev is not None:
                    _plt.switch_backend(prev)
            except Exception:
                pass
    except Exception:
        return fn()


def _with_plot_suppressed(fn):
    """
    Run `fn` while preventing pyplot from opening GUI windows.

    py4DSTEM may call `matplotlib.pyplot.show()` internally even for non-plot
    steps. In a Tk app this can crash. We temporarily replace show/pause with
    no-ops and (best-effort) switch backend to Agg.
    """

    try:
        import matplotlib.pyplot as _plt

        prev_show = getattr(_plt, "show", None)
        prev_pause = getattr(_plt, "pause", None)

        def _noop(*_a, **_k):
            return None

        try:
            _plt.show = _noop  # type: ignore[assignment]
            _plt.pause = _noop  # type: ignore[assignment]
        except Exception:
            pass

        # Best-effort: backend switch (may fail once pyplot is initialized).
        try:
            _plt.switch_backend("Agg")
        except Exception:
            pass

        try:
            return fn()
        finally:
            try:
                if prev_show is not None:
                    _plt.show = prev_show  # type: ignore[assignment]
                if prev_pause is not None:
                    _plt.pause = prev_pause  # type: ignore[assignment]
            except Exception:
                pass
    except Exception:
        return fn()


def _simple_strain_figure(arr: np.ndarray, title: str = "Strain map"):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.asarray(arr, dtype=float), cmap="RdBu_r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _extract_first_axes_image_array(fig) -> np.ndarray:
    for ax in fig.axes:
        if ax.images:
            arr = np.asarray(ax.images[0].get_array(), dtype=float)
            if arr.ndim == 2:
                return arr
    raise RuntimeError("Could not extract a 2D strain map array from the strain figure.")


def compute_line_profile_figure(
    state: WorkflowState,
    label: str,
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: int = 3,
    log: Callable[[str], None] | None = None,
):
    raw = state.strain_raw.get(label)
    S = np.asarray(raw if raw is not None else state.strain_arrays[label])
    S = np.squeeze(S)
    if np.issubdtype(S.dtype, np.number):
        S = _sanitize_strain_values(np.asarray(S))
    # Notebook mapping: 0=εyy, 1=εxx, 2=εxy
    if S.ndim == 3:
        exx = _strain_channel(S, 1)
        eyy = _strain_channel(S, 0)
        exy = _strain_channel(S, 2)
    else:
        exx = _sanitize_strain_values(np.asarray(state.strain_arrays[label], dtype=float))
        eyy = exx
        exy = exx

    w = max(1, int(width))
    d, p_exx = _sample_line_profile(exx, p0, p1, width=w)
    _, p_eyy = _sample_line_profile(eyy, p0, p1, width=w)
    _, p_exy = _sample_line_profile(exy, p0, p1, width=w)

    # Display settings from strain params (percent)
    vr = state.strain_params.get("get_strain", {}).get("vrange", [-2, 2])
    vmin, vmax = float(vr[0]), float(vr[1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].imshow(exx * 100.0, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="upper")
    axes[0].plot([p0[0], p1[0]], [p0[1], p1[1]], color="yellow", linewidth=2)
    axes[0].set_title(f"εxx map + line ({label})")
    axes[1].plot(d, p_exx * 100.0, label="εxx")
    axes[1].plot(d, p_eyy * 100.0, label="εyy")
    axes[1].plot(d, p_exy * 100.0, label="εxy")
    axes[1].set_xlabel("distance (px)")
    axes[1].set_ylabel("strain (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Line profiles (same line, 3 channels)")
    _log(log, f"Computed 3-channel line profile for {label}.")
    return fig, {"distance_px": d, "exx": p_exx, "eyy": p_eyy, "exy": p_exy}


# Distinct colors for multi-line figures (cycled if more lines than colors).
_LINE_COLORS = [
    "crimson", "deepskyblue", "limegreen", "darkorange",
    "purple", "gold", "hotpink", "cyan", "lime", "coral",
]


def compute_multi_line_profiles_figure(
    state: WorkflowState,
    label: str,
    segments: dict,
    width: int = 3,
    strain_component: str = "exx",
    log: Callable[[str], None] | None = None,
):
    """
    Single combined figure for multiple line profiles:
      • Left  — strain map with every line segment drawn in a distinct color
      • Right — strain profile per line, same colors with matching labels
    Parameters
    ----------
    segments : dict  {lid: ((x0,y0),(x1,y1))}  — line-id → (p0, p1)
    strain_component : ``exx`` | ``eyy`` | ``exy`` — which tensor component to profile
    Returns (fig, profile_data_dict)  where profile_data_dict: {lid: {"distance_px", component}}
    """
    comp_key = str(strain_component or "exx").lower()
    ch_map = {"exx": 1, "eyy": 0, "exy": 2}
    ch = ch_map.get(comp_key, 1)
    comp_labels = {"exx": "εxx", "eyy": "εyy", "exy": "εxy"}
    comp_label = comp_labels.get(comp_key, "εxx")

    raw = state.strain_raw.get(label)
    S = np.asarray(raw if raw is not None else state.strain_arrays[label])
    S = np.squeeze(S)
    if np.issubdtype(S.dtype, np.number):
        S = _sanitize_strain_values(np.asarray(S))
    if S.ndim == 3:
        strain_map = _strain_channel(S, ch)
    else:
        strain_map = _sanitize_strain_values(np.asarray(state.strain_arrays[label], dtype=float))

    vr = state.strain_params.get("get_strain", {}).get("vrange", [-2, 2])
    if comp_key in ("exy",):
        vt = state.strain_params.get("get_strain", {}).get("vrange_theta", [-45, 45])
        if vt and len(vt) == 2:
            vr = vt
    vmin, vmax = float(vr[0]), float(vr[1])
    w = max(1, int(width))
    n = len(segments)

    # ── Resolve R-pixel calibration for nm axes ──────────────────────────────
    r_px: float | None = getattr(state, "image_pixel_size", None)
    r_units: str = getattr(state, "image_pixel_units", None) or "nm"
    if r_px is None or r_px <= 0:
        try:
            cal = getattr(getattr(state, "datacube", None), "calibration", None)
            if cal is not None and hasattr(cal, "get_R_pixel_size"):
                v = float(cal.get_R_pixel_size())
                if v > 0:
                    r_px = v
                    r_units = str(cal.get_R_pixel_units() or "nm")
        except Exception:
            pass
    use_nm = r_px is not None and r_px > 0
    H, W = strain_map.shape
    map_extent = [0, W * r_px, H * r_px, 0] if use_nm else None
    dist_scale = r_px if use_nm else 1.0
    dist_label = f"distance ({r_units})" if use_nm else "distance (px)"
    y_scale = 100.0 if comp_key != "exy" else 1.0
    y_units = "%" if comp_key != "exy" else "°"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Left — strain component map
    axes[0].imshow(
        strain_map * y_scale, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="upper",
        **({"extent": map_extent} if map_extent is not None else {}),
    )
    axes[0].set_title(f"{comp_label} — {label} — {n} line{'s' if n != 1 else ''}")
    if use_nm:
        axes[0].set_xlabel(f"x ({r_units})")
        axes[0].set_ylabel(f"y ({r_units})")

    # Right — profiles
    axes[1].set_xlabel(dist_label)
    axes[1].set_ylabel(f"strain ({y_units})" if comp_key != "exy" else "orientation (°)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f"{comp_label} profile{'s' if n != 1 else ''}")

    profile_data: dict = {}
    for i, (lid, seg) in enumerate(sorted(segments.items())):
        p0, p1 = seg
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        d, p_val = _sample_line_profile(strain_map, p0, p1, width=w)
        d_plot = d * dist_scale  # convert to nm if calibrated, else keep px
        # Overlay line on map (coordinates scale with extent)
        px0 = p0[0] * r_px if use_nm else p0[0]
        py0 = p0[1] * r_px if use_nm else p0[1]
        px1 = p1[0] * r_px if use_nm else p1[0]
        py1 = p1[1] * r_px if use_nm else p1[1]
        axes[0].plot(
            [px0, px1], [py0, py1],
            color=color, linewidth=2.5, label=f"L{lid}",
        )
        # Profile curve
        axes[1].plot(d_plot, p_val * y_scale, color=color, label=f"Line {lid}", linewidth=1.8)
        profile_data[lid] = {"distance_px": d, comp_key: p_val}

    if n > 1:
        axes[0].legend(loc="upper right", fontsize=8, framealpha=0.7)
        axes[1].legend(fontsize=8)

    _log(log, f"Combined {n}-line profile figure for {label}.")
    return fig, profile_data


def _sample_line_profile(arr: np.ndarray, p0, p1, width: int = 3):
    x0, y0 = map(float, p0)
    x1, y1 = map(float, p1)
    length = max(2, int(round(np.hypot(x1 - x0, y1 - y0))) + 1)
    xs = np.linspace(x0, x1, length)
    ys = np.linspace(y0, y1, length)
    dx = x1 - x0
    dy = y1 - y0
    norm = np.hypot(dx, dy) or 1.0
    nx = -dy / norm
    ny = dx / norm
    offsets = np.arange(-(width // 2), width // 2 + 1)
    samples = []
    for off in offsets:
        samples.append(_nearest_sample(arr, xs + off * nx, ys + off * ny))
    values = np.nanmean(np.vstack(samples), axis=0)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    distances = np.linspace(0, float(np.hypot(x1 - x0, y1 - y0)), length)
    return distances, values


def _nearest_sample(arr: np.ndarray, xs, ys):
    xi = np.clip(np.rint(xs).astype(int), 0, arr.shape[1] - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, arr.shape[0] - 1)
    return arr[yi, xi]


def _safe_interp(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """1D linear interpolation; returns NaNs if ``x_old`` is invalid or too short."""
    x_new = np.asarray(x_new, dtype=float).ravel()
    x_old = np.asarray(x_old, dtype=float).ravel()
    y_old = np.asarray(y_old, dtype=float).ravel()
    if x_new.size == 0 or x_old.size == 0 or y_old.size == 0:
        return np.full_like(x_new, np.nan, dtype=float)
    if x_old.size != y_old.size:
        n = min(int(x_old.size), int(y_old.size))
        x_old = x_old[:n]
        y_old = y_old[:n]
    else:
        n = int(x_old.size)
    if n < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    order = np.argsort(x_old)
    x_old = x_old[order]
    y_old = y_old[order]
    ux, idx = np.unique(x_old, return_index=True)
    y_u = y_old[idx]
    if ux.size < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    return np.interp(x_new, ux, y_u, left=np.nan, right=np.nan)


def _jsonify_line_segment(seg: Any) -> list[list[float]] | None:
    if not isinstance(seg, (tuple, list)) or len(seg) != 2:
        return None
    p0, p1 = seg
    try:
        return [list(map(float, p0)), list(map(float, p1))]
    except Exception:
        return None


def serialize_segment_dict(d: Any) -> dict[str, list[list[float]]] | None:
    """``fixed_line_profiles_px`` and inner line dicts → JSON-safe ``{"1":[[x0,y0],[x1,y1]], ...}``."""
    if not isinstance(d, dict) or not d:
        return None
    out: dict[str, list[list[float]]] = {}
    for lid, seg in d.items():
        try:
            k = int(lid)
        except Exception:
            continue
        js = _jsonify_line_segment(seg)
        if js is None:
            continue
        out[str(k)] = js
    return out or None


def deserialize_segment_dict(data: Any) -> dict[int, tuple[tuple[float, float], tuple[float, float]]] | None:
    if not isinstance(data, dict) or not data:
        return None
    out: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    for lid_s, seg in data.items():
        try:
            lid = int(lid_s)
        except Exception:
            continue
        if not isinstance(seg, (list, tuple)) or len(seg) != 2:
            continue
        p0, p1 = seg
        try:
            out[lid] = (tuple(map(float, p0)), tuple(map(float, p1)))
        except Exception:
            continue
    return out or None


def serialize_line_profiles_nested(d: Any) -> dict[str, dict[str, list[list[float]]]] | None:
    """``line_profiles_px`` / ``stress_line_profiles_px``: label → line-id → segment."""
    if not isinstance(d, dict) or not d:
        return None
    out: dict[str, dict[str, list[list[float]]]] = {}
    for label, inner in d.items():
        s = serialize_segment_dict(inner)
        if s is not None:
            out[str(label)] = s
    return out or None


def deserialize_line_profiles_nested(
    data: Any,
) -> dict[str, dict[int, tuple[tuple[float, float], tuple[float, float]]]]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[int, tuple[tuple[float, float], tuple[float, float]]]] = {}
    for label, inner in data.items():
        dec = deserialize_segment_dict(inner)
        if dec:
            out[str(label)] = dec
    return out


def _stress_unit_token(unit: Any) -> str:
    """Plain unit label for CSV row 2 (no parentheses or spaces)."""
    return str(unit or "GPa").strip().replace(" ", "").replace("(", "").replace(")", "")


def _roi_csv_label(tag: str) -> str:
    """Human-readable ROI row for Origin-style CSV headers."""
    t = str(tag or "").strip()
    if t in ("withoutROI", "without_roi"):
        return "W/O ROI"
    if t in ("withROI", "with_roi"):
        return "W/ ROI"
    return t


def _write_csv_originlab_four_headers(
    path: Path,
    col_meta: list[tuple[str, ...]],
    columns_data: list[np.ndarray],
) -> None:
    """
    Write numeric CSV with five leading rows for OriginLab:
    1) quantity, 2) units, 3) tensor/component, 4) ROI, 5) scan / file name.
    Each ``col_meta`` entry is ``(long_name, unit, tensor, roi, scan)``; use "" for unused cells.
    Legacy 4-tuples ``(long_name, unit, tensor, roi)`` are still accepted (scan row empty).
    """
    import csv

    if len(col_meta) != len(columns_data):
        raise ValueError("col_meta and columns_data length mismatch")
    rows5: list[tuple[str, str, str, str, str]] = []
    for m in col_meta:
        if len(m) >= 5:
            rows5.append((str(m[0]), str(m[1]), str(m[2]), str(m[3]), str(m[4])))
        elif len(m) == 4:
            rows5.append((str(m[0]), str(m[1]), str(m[2]), str(m[3]), ""))
        else:
            raise ValueError(f"col_meta entry must have 4 or 5 fields, got {m!r}")
    n = int(columns_data[0].size) if columns_data else 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([m[0] for m in rows5])
        w.writerow([m[1] for m in rows5])
        w.writerow([m[2] for m in rows5])
        w.writerow([m[3] for m in rows5])
        w.writerow([m[4] for m in rows5])
        for i in range(n):
            w.writerow(
                [
                    float(columns_data[j][i]) if i < len(columns_data[j]) else ""
                    for j in range(len(columns_data))
                ]
            )


def _write_consolidated_line_profile_csv(
    state: WorkflowState,
    data_dir: Path,
    line_id: int,
    log: Callable[[str], None] | None = None,
) -> Path | None:
    """
    One CSV per scan line (L1 or L2) merging every available profile type:
    strain without ROI, strain with ROI, stress without ROI, stress with ROI.

    Header layout (four rows) targets Origin import: long name, units, tensor id,
    ROI flag. Strain columns are **percent**; stress units come from the stress step.
    """
    lid = int(line_id)
    strain_wo = state.strain_raw.get(f"line_profile_data_without_roi_{lid}")
    strain_wr = state.strain_raw.get(f"line_profile_data_with_roi_{lid}")
    stress_wo = state.stress_line_profile_data.get(f"stress_line_profile_without_roi_{lid}")
    stress_wr = state.stress_line_profile_data.get(f"stress_line_profile_with_roi_{lid}")

    def _dist(dat: Any) -> np.ndarray | None:
        if not isinstance(dat, dict):
            return None
        d = dat.get("distance_px")
        if d is None:
            return None
        a = np.asarray(d, dtype=float).ravel()
        return a if a.size > 0 else None

    candidates = (_dist(strain_wo), _dist(strain_wr), _dist(stress_wo), _dist(stress_wr))
    master = next((c for c in candidates if c is not None), None)
    if master is None:
        return None

    meta: list[tuple[str, str, str, str, str]] = [("distance", "px", "", "", "")]
    cols: list[np.ndarray] = [master]
    scan_stem = ""
    try:
        if state.raw_mib_path is not None:
            scan_stem = Path(state.raw_mib_path).stem
    except Exception:
        pass

    def _add_strain(roi_tag: str, dat: Any) -> None:
        if not isinstance(dat, dict):
            return
        d = _dist(dat)
        if d is None:
            return
        roi_lbl = _roi_csv_label(roi_tag)
        for key, tensor in (("exx", "eps_xx"), ("eyy", "eps_yy"), ("exy", "eps_xy")):
            if key not in dat:
                continue
            y = np.asarray(dat[key], dtype=float).ravel() * 100.0
            meta.append(("Strain", "%", tensor, roi_lbl, scan_stem))
            cols.append(_safe_interp(master, d, y))

    _add_strain("withoutROI", strain_wo)
    _add_strain("withROI", strain_wr)

    def _add_stress(roi_tag: str, dat: Any) -> None:
        if not isinstance(dat, dict):
            return
        d = _dist(dat)
        if d is None:
            return
        unit = _stress_unit_token(dat.get("unit"))
        roi_lbl = _roi_csv_label(roi_tag)
        for key in ("sigma_xx", "sigma_xy", "sigma_yy"):
            if key not in dat:
                continue
            y = np.asarray(dat[key], dtype=float).ravel()
            if unit.upper() == "GPA":
                y = y * 1000.0
                col_unit = "MPa"
            else:
                col_unit = unit
            meta.append(("Stress", col_unit, key, roi_lbl, scan_stem))
            cols.append(_safe_interp(master, d, y))

    _add_stress("withoutROI", stress_wo)
    _add_stress("withROI", stress_wr)

    if len(cols) <= 1:
        return None

    fname = f"00_line_profile_line{lid}_consolidated_all_sources.csv"
    path = data_dir / fname
    _write_csv_originlab_four_headers(path, meta, cols)
    _log(log, f"Consolidated line {lid} profile CSV: {path}")
    return path


def stamp_figure_scan_title(
    fig,
    scan_name: str,
    subtitle: str = "",
    *,
    y: float = 0.98,
) -> None:
    """Burn scan identity into a figure before save/display (batch export)."""
    if fig is None or not hasattr(fig, "suptitle"):
        return
    title = str(scan_name).strip() or "scan"
    if subtitle:
        title = f"{title} — {subtitle}"
    try:
        fig.suptitle(title, fontsize=11, y=y)
    except Exception:
        pass


def save_final_figures_step(
    state: WorkflowState,
    output_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
    ui_extras: dict[str, Any] | None = None,
) -> list[Path]:
    # New structure:
    # gui_results/<scan_name>/{figures,data}
    if output_dir is None:
        base = (state.raw_mib_path.parent if state.raw_mib_path is not None else Path.cwd()) / "gui_results"
    else:
        base = Path(output_dir).expanduser()
    scan_name = state.raw_mib_path.stem if state.raw_mib_path is not None else "results"
    out_dir = base / scan_name
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(s: str) -> str:
        import re

        s = re.sub(r"\s+", "_", str(s).strip())
        s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
        return s or "figure"

    name_map = {
        "without_roi": "01_strain_mapping_without_ROI.png",
        "line_profile_without_roi_1": "02_line_profile_without_ROI_line1.png",
        "line_profile_without_roi_2": "03_line_profile_without_ROI_line2.png",
        "with_roi": "03_strain_mapping_WITH_ROI.png",
        "line_profile_with_roi_1": "04_line_profile_WITH_ROI_line1.png",
        "line_profile_with_roi_2": "05_line_profile_WITH_ROI_line2.png",
        "compare_full_roi_overlay": "10_compare_full_exx_with_roi_overlay.png",
    }
    saved: list[Path] = []
    for key, filename in name_map.items():
        fig = state.strain_figures.get(key)
        if fig is None:
            continue
        path = fig_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        saved.append(path)

    for key, filename in (
        ("without_roi", "11_stress_maps_without_ROI.png"),
        ("with_roi", "12_stress_maps_with_ROI.png"),
    ):
        sfig = state.stress_figures.get(key)
        if sfig is None:
            continue
        path = fig_dir / filename
        sfig.savefig(path, dpi=300, bbox_inches="tight")
        saved.append(path)

    for key in ("without_roi", "with_roi"):
        st = state.stress_tensors_pa.get(key)
        if st is None:
            continue
        np.savez_compressed(
            data_dir / f"stress_tensors_{key}_Pa.npz",
            sigma_xx=np.asarray(st["sigma_xx"]),
            sigma_yy=np.asarray(st["sigma_yy"]),
            sigma_xy=np.asarray(st["sigma_xy"]),
        )
        meta = state.stress_meta.get(key)
        if meta:
            try:
                import json

                (data_dir / f"stress_meta_{key}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception:
                pass

    # Virtual-image grid (ADF, BF, DP mean, DP max): regenerate so save works even if Step 3 tab was closed.
    try:
        try:
            from .viewer import build_virtual_image_figure
        except ImportError:
            from viewer import build_virtual_image_figure

        if state.visualcube is not None or state.precomputed_h5_path:
            load_virtual_images_step(state, log=log)
            sg = scan_name + " — ADF, BF, DP mean, DP max"
            vfig = build_virtual_image_figure(state, title=sg)
            vpath = fig_dir / "14_virtual_images_ADF_BF_DP_grid.png"
            vfig.savefig(vpath, dpi=300, bbox_inches="tight")
            saved.append(vpath)
    except Exception as exc:
        _log(log, f"Note: virtual images grid not saved (need .h5 / Step 3 data): {exc}")

    # Step 12 basis preview (choose_basis_vectors panels).
    b_idx = 1
    for bf in list(state.basis_preview_figures or []):
        if bf is None or not hasattr(bf, "savefig"):
            continue
        try:
            sub = f"Strain basis (panel {b_idx})"
            stamp_figure_scan_title(bf, scan_name, sub, y=1.0 if len(getattr(bf, "axes", [])) >= 2 else 0.98)
            safe_stem = _safe_name(scan_name)
            bp = fig_dir / f"15_{safe_stem}_step12_basis_preview_{b_idx}.png"
            bf.savefig(bp, dpi=300, bbox_inches="tight")
            saved.append(bp)
            b_idx += 1
        except Exception as exc:
            _log(log, f"Warning: could not save Step 12 basis preview #{b_idx}: {exc}")

    # Save ALL UI figures shown in tabs (ADF/BF, ROI, origin correction, q-pixel overlay,
    # ellipse (if fitted), basis preview figures, etc.)
    def _ellipse_is_valid() -> bool:
        if state.p_ellipse is None:
            return False
        try:
            cal = getattr(getattr(state.braggpeaks, "calibration", None), "calstate", None)
            if isinstance(cal, dict):
                v = cal.get("ellipse", True)
                return bool(v)
        except Exception:
            pass
        return True

    already = {p.name for p in saved}
    idx = 20
    # Omit intermediate Step 11 pixel-size *sweep* panels (keep overlay, refit, summary).
    _skip_title_prefixes = ("Step 11 Q test ", "Step 12 basis ")
    for title, fig in (state.figures or {}).items():
        if fig is None or not hasattr(fig, "savefig"):
            continue
        t = str(title)
        if t.startswith(_skip_title_prefixes):
            continue
        if "ellipse" in t.lower() and not _ellipse_is_valid():
            continue
        fname = f"{idx:02d}_{_safe_name(t)}.png"
        if fname in already:
            continue
        try:
            stamp_figure_scan_title(fig, scan_name, t)
            path = fig_dir / fname
            fig.savefig(path, dpi=300, bbox_inches="tight")
            saved.append(path)
            already.add(fname)
            idx += 1
        except Exception as exc:
            _log(log, f"Warning: could not save figure '{title}': {exc}")

    # Save line-profile CSVs (for external plotting; OriginLab-friendly 4 header rows)
    try:
        csv_keys = [
            ("line_profile_data_without_roi_1", "06_line_profile_without_ROI_line1.csv"),
            ("line_profile_data_without_roi_2", "07_line_profile_without_ROI_line2.csv"),
            ("line_profile_data_with_roi_1", "08_line_profile_WITH_ROI_line1.csv"),
            ("line_profile_data_with_roi_2", "09_line_profile_WITH_ROI_line2.csv"),
        ]
        for key, filename in csv_keys:
            data = state.strain_raw.get(key)
            if not isinstance(data, dict):
                continue
            roi_tag = "withoutROI" if "without_roi" in key else "withROI"
            d = np.asarray(data["distance_px"], dtype=float)
            exx = np.asarray(data["exx"], dtype=float) * 100.0
            eyy = np.asarray(data["eyy"], dtype=float) * 100.0
            exy = np.asarray(data["exy"], dtype=float) * 100.0
            path = data_dir / filename
            scan_stem = Path(state.raw_mib_path).stem if state.raw_mib_path else ""
            roi_lbl = _roi_csv_label(roi_tag)
            col_meta: list[tuple[str, str, str, str, str]] = [
                ("distance", "px", "", "", ""),
                ("Strain", "%", "eps_xx", roi_lbl, scan_stem),
                ("Strain", "%", "eps_yy", roi_lbl, scan_stem),
                ("Strain", "%", "eps_xy", roi_lbl, scan_stem),
            ]
            _write_csv_originlab_four_headers(path, col_meta, [d, exx, eyy, exy])
            saved.append(path)

        # Stress line-profile CSVs (σ_xx and σ_yy along each line; unit set by GUI when computed)
        for label in ("without_roi", "with_roi"):
            roi_tag = "withoutROI" if label == "without_roi" else "withROI"
            for lid in (1, 2):
                sdat = state.stress_line_profile_data.get(f"stress_line_profile_{label}_{lid}")
                if not isinstance(sdat, dict):
                    continue
                d = np.asarray(sdat.get("distance_px"), dtype=float)
                unit = _stress_unit_token(sdat.get("unit"))
                col_unit = "MPa" if unit.upper() == "GPA" else unit
                scale = 1000.0 if unit.upper() == "GPA" else 1.0
                stress_cols: list[np.ndarray] = [d]
                scan_stem = Path(state.raw_mib_path).stem if state.raw_mib_path else ""
                roi_lbl = _roi_csv_label(roi_tag)
                col_meta = [("distance", "px", "", "", "")]
                for key in ("sigma_xx", "sigma_xy", "sigma_yy"):
                    if key not in sdat:
                        continue
                    y = np.asarray(sdat[key], dtype=float).ravel() * scale
                    col_meta.append(("Stress", col_unit, key, roi_lbl, scan_stem))
                    stress_cols.append(y)
                if len(stress_cols) <= 1:
                    continue
                path = data_dir / f"13_stress_line_profile_{label}_line{lid}_{col_unit}.csv"
                _write_csv_originlab_four_headers(path, col_meta, stress_cols)
                saved.append(path)

        for lid in (1, 2):
            try:
                p = _write_consolidated_line_profile_csv(state, data_dir, lid, log=log)
                if p is not None:
                    saved.append(p)
            except Exception as exc:
                _log(log, f"Warning: could not write consolidated line profile CSV (line {lid}): {exc}")
    except Exception as exc:
        _log(log, f"Warning: could not save line-profile CSVs: {exc}")
    _log(log, f"Saved outputs to: {out_dir}")
    # Save parameters snapshot for reproducibility
    try:
        import json

        sb = state.strain_basis_params or {}
        _cal = getattr(getattr(state, "braggpeaks", None), "calibration", None)
        _qr = sb.get("qr_rotation")
        if _qr is None and _cal is not None:
            _qr = getattr(_cal, "QR_rotation", None)
        _qf = sb.get("qr_flip")
        if _qf is None and _cal is not None:
            _qf = getattr(_cal, "QR_flip", None)

        roi_box: list[int] | None = None
        rect = getattr(state, "strain_roi_rect", None)
        if rect is not None and len(rect) == 4:
            roi_box = [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]
        elif state.strain_roi_mask is not None:
            try:
                arr = np.asarray(state.strain_roi_mask, dtype=bool)
                ys, xs = np.where(arr)
                if ys.size:
                    roi_box = [int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1]
            except Exception:
                pass

        params: dict[str, Any] = {
            "q_pixel_size": state.q_pixel_size,
            "q_pixel_units": state.q_pixel_units,
            "detect_params": state.detect_params,
            "disk_preview_params": state.disk_preview_params,
            "strain_basis_params": state.strain_basis_params,
            "strain_params": state.strain_params,
            "qr_rotation": _qr,
            "qr_flip": _qf,
            "roi_bounds": state.roi_bounds,
            "origin": {
                "center_guess": list(state.center_guess) if state.center_guess is not None else None,
            },
            "sampling": state.origin_sampling,
            "strain_roi_pixel_box": roi_box,
            "strain_external_reference_g12": (
                [
                    [float(state.strain_external_reference_g12[0][0]), float(state.strain_external_reference_g12[0][1])],
                    [float(state.strain_external_reference_g12[1][0]), float(state.strain_external_reference_g12[1][1])],
                ]
                if getattr(state, "strain_external_reference_g12", None) is not None
                and len(state.strain_external_reference_g12) == 2
                else None
            ),
            "strain_use_external_reference": bool(getattr(state, "strain_use_external_reference", False)),
            "line_profiles_px": serialize_line_profiles_nested(getattr(state, "line_profiles_px", {}) or {}),
            "fixed_line_profiles_px": serialize_segment_dict(getattr(state, "fixed_line_profiles_px", None)),
            "stress_line_profiles_px": serialize_line_profiles_nested(
                getattr(state, "stress_line_profiles_px", {}) or {}
            ),
        }
        if ui_extras:
            params.update(ui_extras)
        (data_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
    except Exception as exc:
        _log(log, f"Warning: could not write params.json: {exc}")
    return saved
