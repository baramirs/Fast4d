"""Tests for EMD/HDF5 DataCube datapath discovery (tutorial-style files)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SIM_H5 = Path(
    r"C:\Users\jtapiaca.ASURITE\AI Projects\GPA\Testers"
    r"\calibrationData_simulatedAuNanoplatelet_binned_v14.h5"
)


def test_discover_datacube_paths_synthetic(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    import pipeline as pl

    path = tmp_path / "demo_AuNanoplatelet_scan.h5"
    with h5py.File(path, "w") as f:
        root = f.create_group("4DSTEM_simulation")
        root.attrs["emd_group_type"] = "root"
        for name, shape in (
            ("4DSTEM_polyAu", (2, 2, 4, 4)),
            ("4DSTEM_AuNanoplatelet", (3, 3, 4, 4)),
        ):
            g = root.create_group(name)
            g.attrs["emd_group_type"] = "array"
            g.attrs["python_class"] = "DataCube"
            g.create_dataset("data", data=np.zeros(shape, dtype=np.uint16))

    paths = pl._discover_emd_datacube_paths(path)
    assert "4DSTEM_simulation/4DSTEM_AuNanoplatelet" in paths
    assert "4DSTEM_simulation/4DSTEM_polyAu" in paths

    ranked = pl._candidate_datapaths(path)
    assert ranked[0] == "4DSTEM_simulation/4DSTEM_AuNanoplatelet"


@pytest.mark.skipif(not SIM_H5.is_file(), reason="tutorial sim H5 not present")
def test_load_sim_aunanoplatelet_prefers_matching_cube():
    import pipeline as pl

    logs: list[str] = []
    cube = pl._load_emd_h5_datacube(SIM_H5, log=logs.append)
    assert tuple(cube.shape) == (100, 84, 125, 125)
    joined = " ".join(logs)
    assert "AuNanoplatelet" in joined
    name = str(getattr(cube, "name", "") or "")
    assert "AuNanoplatelet" in name or "AuNanoplatelet" in joined
    # Prefer matching cube over polyAu regardless of tree=True/False flakiness.
    assert "polyAu" not in joined.split("Loaded DataCube")[-1]
