"""
Fast4D-side incremental Bragg-peak writer.

py4DSTEM's own ``PointListArray`` (emdfile/classes/pointlistarray.py:55-90)
eagerly allocates a full list-of-lists covering every scan position before any
disks are found, and has no streaming write path (confirmed in
MEMORY_ARCHITECTURE_REPORT.md Part IV). This module writes one HDF5 dataset
per scan position as detection proceeds, so peak storage never requires the
whole scan's peak list to be resident in RAM at once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np


class StreamingBraggWriter:
    """Writes detected peaks to ``path`` one scan position at a time.

    Layout: ``/r_shape`` = (R_Ny, R_Nx); ``/peaks/{ry}_{rx}`` = (n_peaks, 3)
    array of (qx, qy, intensity), one dataset per position that was written.
    """

    def __init__(self, path: str | Path, r_shape: tuple[int, int]) -> None:
        self._path = Path(path)
        self._r_shape = r_shape
        self._file: h5py.File | None = None

    def __enter__(self) -> "StreamingBraggWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "w")
        self._file.create_dataset("r_shape", data=np.array(self._r_shape, dtype=np.int64))
        self._file.create_group("peaks")
        return self

    def write(self, ry: int, rx: int, qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray) -> None:
        if self._file is None:
            raise RuntimeError("StreamingBraggWriter used outside a `with` block.")
        qx = np.asarray(qx, dtype=np.float64)
        qy = np.asarray(qy, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)
        n = qx.shape[0]
        arr = np.empty((n, 3), dtype=np.float64)
        arr[:, 0], arr[:, 1], arr[:, 2] = qx, qy, intensity
        self._file["peaks"].create_dataset(f"{ry}_{rx}", data=arr)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def detect_braggpeaks_streaming(
    datacube,
    template,
    out_path: str | Path,
    *,
    batch_size: int = 64,
    log: Callable[[str], None] | None = None,
    **detect_kwargs,
) -> Path:
    """Detect Bragg disks in small position-batches and stream results to disk.

    Uses the same public, position-scoped call Fast4D already makes for its
    6-point preview (``pipeline.py:1344,1371``: ``datacube.find_Bragg_disks(
    data=(<x-positions>, <y-positions>), template=..., **kwargs)``), just
    looped over the whole scan in ``batch_size``-position chunks instead of
    either 6 positions (preview) or the whole scan at once
    (``compute_braggpeaks_step``, ``pipeline.py:1416``). Only ``batch_size``
    positions' worth of detected peaks are ever held in RAM at a time.

    Axis-order note: ``DataCube.find_Bragg_disks`` unpacks a 2-tuple ``data``
    as ``dc, rx, ry = data[0], data[1], data[2]`` internally (after prepending
    ``self``) and indexes ``dc.data[rx, ry, :, :]``
    (``braggvectors/diskdetection.py:203,220``), matching the axis order the
    full-datacube path itself loops in
    (``for rx, ry in tqdmnd(datacube.R_Nx, datacube.R_Ny)``,
    ``braggvectors/diskdetection.py:479-481``). So the first array in
    ``data=(...)`` must range over ``R_Nx`` and the second over ``R_Ny``; this
    function passes ``data=(rxs, rys)`` accordingly (verified by a throwaway
    script showing this batched call's output matches
    ``BraggVectors.raw[rx, ry]`` from the full-scan call, position-by-position).

    Accessor note (py4DSTEM==0.14.19): when ``data`` is a 2-tuple of
    position-index arrays, ``DataCube.find_Bragg_disks`` does NOT return a
    ``BraggVectors`` instance the way the no-``data`` full-scan call does.
    It returns a plain ``list[QPoints]``, one entry per requested position, in
    the same order the position arrays were given (confirmed by reading
    ``braggvectors/diskdetection.py`` and by a throwaway script that compared
    this batched call's output position-by-position against the full-scan
    ``BraggVectors.raw[rx, ry]`` accessor -- they matched exactly). Each
    ``QPoints`` instance exposes fields directly as ``.qx``, ``.qy``,
    ``.intensity`` (``py4DSTEM/data/qpoints.py``), so no ``get_vectors``/
    ``.cal``/``.raw`` fallback chain is needed or applicable here.
    """
    out_path = Path(out_path)
    r_nx, r_ny = datacube.R_Nx, datacube.R_Ny
    all_positions = [(rx, ry) for rx in range(r_nx) for ry in range(r_ny)]

    with StreamingBraggWriter(out_path, r_shape=(r_nx, r_ny)) as writer:
        for start in range(0, len(all_positions), batch_size):
            batch = all_positions[start:start + batch_size]
            rxs = [p[0] for p in batch]
            rys = [p[1] for p in batch]
            results = datacube.find_Bragg_disks(data=(rxs, rys), template=template, **detect_kwargs)
            for (rx, ry), pl in zip(batch, results):
                writer.write(ry, rx, qx=np.asarray(pl.qx), qy=np.asarray(pl.qy),
                             intensity=np.asarray(pl.intensity))
            if log is not None:
                log(f"Streaming Bragg detection: {min(start + batch_size, len(all_positions))}/{len(all_positions)} positions done.")
    return out_path


def read_streamed_peaks(path: str | Path, ry: int, rx: int) -> dict:
    """Read one scan position's peaks from a streamed file — O(1) HDF5 lookup,
    never loads other positions' peaks into RAM."""
    with h5py.File(path, "r") as f:
        key = f"{ry}_{rx}"
        if key not in f["peaks"]:
            return {"qx": np.array([]), "qy": np.array([]), "intensity": np.array([])}
        arr = f["peaks"][key][()]
        return {"qx": arr[:, 0], "qy": arr[:, 1], "intensity": arr[:, 2]}
