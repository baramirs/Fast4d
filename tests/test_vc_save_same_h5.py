"""ADF/BF should append into the same tutorial EMD .h5 (not a new sidecar)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_vc_save_appends_virtual_images_to_same_emd(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    py4DSTEM = pytest.importorskip("py4DSTEM")

    import engine as E
    import pipeline as pl

    path = tmp_path / "calibrationData_simulatedAuNanoplatelet_demo.h5"

    au = py4DSTEM.DataCube(
        data=np.ones((2, 2, 8, 8), dtype=np.uint16), name="4DSTEM_AuNanoplatelet"
    )
    # Force the EMD root name to match the tutorial layout.
    au.root.name = "4DSTEM_simulation"
    py4DSTEM.save(str(path), au, mode="w", tree=True)

    poly = py4DSTEM.DataCube(
        data=np.full((2, 2, 8, 8), 2, dtype=np.uint16), name="4DSTEM_polyAu"
    )
    poly.root.name = "4DSTEM_simulation"
    # Graft sibling under the same root name via append.
    py4DSTEM.save(str(path), poly, mode="a", tree=True)

    ranked = pl._candidate_datapaths(path)
    assert any("AuNanoplatelet" in str(p) for p in ranked)

    cube = pl._load_emd_h5_datacube(
        path, emd_datapath="4DSTEM_simulation/4DSTEM_AuNanoplatelet"
    )
    assert "AuNanoplatelet" in str(getattr(cube, "_fast4d_emd_datapath", ""))

    for key, arr in (
        ("dp_mean", np.ones((8, 8), dtype=np.float32)),
        ("annular_dark_field", np.arange(4, dtype=np.float32).reshape(2, 2)),
        ("bright_field", np.arange(4, dtype=np.float32).reshape(2, 2)[::-1]),
    ):
        cube.add_to_tree(py4DSTEM.Array(data=arr, name=key))

    sc = E.Scan(name="sim", raw_path=str(path), h5_path=str(path))
    st = sc.ensure_state()
    st.datacube = cube
    st.emd_datapath = cube._fast4d_emd_datapath

    out = E.vc_save_h5(sc, str(path))
    assert Path(out).resolve() == path.resolve()

    with h5py.File(path, "r") as f:
        sim = f["4DSTEM_simulation"]
        assert "4DSTEM_polyAu" in sim  # sibling preserved
        au_g = sim["4DSTEM_AuNanoplatelet"]
        assert "annular_dark_field" in au_g
        assert "bright_field" in au_g
        assert "dp_mean" in au_g

    visual = pl._load_visualcube_from_h5(path)
    adf = pl._read_tree(visual, "annular_dark_field")
    arr = np.asarray(adf.data if hasattr(adf, "data") else adf)
    assert arr.shape == (2, 2)
