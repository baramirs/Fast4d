"""ROI-mean diffraction pattern → native-resolution TIFF for CrystBox."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_q_Ainv_to_nminv():
    from tools.roi_mean_dp import q_Ainv_to_nminv

    assert q_Ainv_to_nminv(0.021572956) == pytest.approx(0.21572956)
    assert q_Ainv_to_nminv(0.1) == pytest.approx(1.0)


def test_mean_dp_keeps_detector_shape():
    from tools.roi_mean_dp import mean_dp_from_4d, roi_to_slices

    rng = np.random.default_rng(0)
    cube = rng.integers(0, 100, size=(8, 10, 32, 28), dtype=np.uint16)
    sl = roi_to_slices({"xMin": 2, "xMax": 5, "yMin": 1, "yMax": 4}, (8, 10))
    dp = mean_dp_from_4d(cube, sl)
    assert dp.shape == (32, 28)
    assert dp.dtype == np.float64
    expect = cube[1:4, 2:5].astype(np.float64).mean(axis=(0, 1))
    np.testing.assert_allclose(dp, expect)


def test_empty_roi_raises():
    from tools.roi_mean_dp import mean_dp_from_4d, roi_to_slices

    cube = np.zeros((4, 4, 8, 8), dtype=np.uint16)
    with pytest.raises(ValueError, match="empty"):
        sl = roi_to_slices({"xMin": 2, "xMax": 2, "yMin": 1, "yMax": 3}, (4, 4))
        mean_dp_from_4d(cube, sl)


def test_uint16_export_same_shape_and_tiff(tmp_path: Path):
    from tools.roi_mean_dp import prepare_dp_array, write_dp_tiff

    dp = np.linspace(0.0, 10.0, 16 * 12, dtype=np.float64).reshape(16, 12)
    out16 = prepare_dp_array(dp, dtype="uint16")
    assert out16.shape == (16, 12)
    assert out16.dtype == np.uint16
    assert out16.max() == 65535

    tif = tmp_path / "roi_dp.tif"
    meta = write_dp_tiff(
        tif, dp,
        q_Ainv_per_px=0.02,
        roi={"xMin": 1, "xMax": 3, "yMin": 0, "yMax": 2},
        n_patterns=4,
        source="scan.mib",
    )
    assert tif.is_file()
    assert tif.stat().st_size > 0
    assert meta["shape"] == [16, 12]
    assert meta["q_nminv_per_px"] == pytest.approx(0.2)
    sidecar = tif.with_suffix(".json")
    assert sidecar.is_file()


def test_virtual_dp_export_name():
    from tools.roi_mean_dp import virtual_dp_export_name

    assert virtual_dp_export_name("Scan01", "dp_max") == "Scan01_DP_max.tif"
    assert virtual_dp_export_name("Scan01.h5", "dp_mean") == "Scan01_DP_mean.tif"
    assert virtual_dp_export_name(Path("Scan06_braggpeaks.h5"), "dp_max") == "Scan06_DP_max.tif"


def test_collect_sidecar_h5_skips_braggpeaks(tmp_path: Path):
    from tools.roi_mean_dp import collect_sidecar_h5

    (tmp_path / "Scan01.h5").write_bytes(b"x")
    (tmp_path / "Scan01_braggpeaks.h5").write_bytes(b"x")
    (tmp_path / "Scan02.h5").write_bytes(b"x")
    names = [p.name for p in collect_sidecar_h5(tmp_path)]
    assert names == ["Scan01.h5", "Scan02.h5"]


def test_export_sidecar_dps_writes_named_tiffs(tmp_path: Path):
    import h5py
    from tools.roi_mean_dp import export_sidecar_dps, read_sidecar_dps

    src = tmp_path / "Scan01.h5"
    mean = np.arange(12, dtype=np.float32).reshape(3, 4)
    mx = (mean + 10).reshape(3, 4)
    with h5py.File(src, "w") as f:
        f.create_dataset("datacube_root/dp_mean/data", data=mean)
        f.create_dataset("datacube_root/dp_max/data", data=mx)
        f.create_dataset("datacube_root/annular_dark_field/data",
                         data=np.ones((8, 8), dtype=np.float32))

    got = read_sidecar_dps(src)
    assert set(got) == {"dp_mean", "dp_max"}
    assert got["dp_mean"].shape == (3, 4)

    out = tmp_path / "out"
    written = export_sidecar_dps(src, out, q_Ainv_per_px=0.02)
    names = sorted(p.name for p in written)
    assert names == ["Scan01_DP_max.tif", "Scan01_DP_mean.tif"]
    assert (out / "Scan01_DP_mean.tif").is_file()
    assert (out / "Scan01_DP_max.json").is_file()
