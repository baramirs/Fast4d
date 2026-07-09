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
