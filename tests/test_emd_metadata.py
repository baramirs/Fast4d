"""Extract STEM vs camera pixel-size metadata from EMD/HDF5 without loading the cube."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _write_velox_emd(path: Path, *, pixel_m: float = 1.015e-10) -> Path:
    """Minimal Thermo Fisher Velox-style EMD: Image metadata JSON, no 4D payload."""
    meta = {
        "BinaryResult": {
            "PixelSize": {"width": str(pixel_m), "height": str(pixel_m)},
            "Detector": "HAADF",
        },
        "Instrument": {"Manufacturer": "FEI", "InstrumentModel": "Titan"},
        "Optics": {"AccelerationVoltage": "300000", "ProbeMode": "STEM"},
    }
    blob = json.dumps(meta).encode("utf-8")
    with h5py.File(path, "w") as f:
        img = f.create_group("Data/Image/img0")
        img.create_dataset("Data", data=np.zeros((8, 8), dtype=np.uint16))
        img.create_dataset("Metadata", data=np.frombuffer(blob, dtype=np.uint8))
        f.create_dataset("Version", data=np.array(b"Velox", dtype="S"))
    return path


def test_velox_pixel_size_is_stem_r_not_merlin(tmp_path: Path):
    from tools.emd_metadata import extract_emd_metadata

    path = _write_velox_emd(tmp_path / "scan.emd")
    report = extract_emd_metadata(path)

    assert report["kind"] == "velox"
    assert report["loaded_4d_payload"] is False
    rec = report["recommendation"]
    assert rec is not None
    assert rec["role"] == "R"
    assert rec["pm_per_px"] == pytest.approx(101.5, rel=1e-6)
    assert rec["nm_per_px"] == pytest.approx(0.1015, rel=1e-6)
    assert rec["source_kind"] == "stem_image"
    assert "HAADF" in rec["note"]
    assert rec["role"] == "R"
    assert rec["nm_per_px"] == pytest.approx(0.1015, rel=1e-6)


def test_py4dstem_r_and_q_pixels_are_separated(tmp_path: Path):
    from tools.emd_metadata import extract_emd_metadata

    path = tmp_path / "cube.h5"
    with h5py.File(path, "w") as f:
        root = f.create_group("4DSTEM_experiment")
        root.attrs["emd_group_type"] = "root"
        g = root.create_group("datacube")
        g.attrs["emd_group_type"] = "array"
        g.attrs["python_class"] = "DataCube"
        g.create_dataset("data", data=np.zeros((2, 2, 4, 4), dtype=np.uint8))
        cal = g.create_group("calibration")
        cal.attrs["R_pixel_size"] = 0.1015
        cal.attrs["R_pixel_units"] = "nm"
        cal.attrs["Q_pixel_size"] = 0.0137
        cal.attrs["Q_pixel_units"] = "A^-1"

    report = extract_emd_metadata(path)
    roles = {h["role"] for h in report["hits"]}
    assert "R" in roles
    assert "Q" in roles
    rec = report["recommendation"]
    assert rec["role"] == "R"
    assert rec["nm_per_px"] == pytest.approx(0.1015)


def test_does_not_read_large_datasets(tmp_path: Path):
    from tools.emd_metadata import extract_emd_metadata

    path = tmp_path / "huge.emd"
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("Data/big", data=np.zeros((64, 64, 32, 32), dtype=np.uint16))
        ds.attrs["R_pixel_size"] = 0.2
        ds.attrs["R_pixel_units"] = "nm"
    report = extract_emd_metadata(path)
    assert report["loaded_4d_payload"] is False
    assert report["hits"], "attrs on the large dataset must still be readable"
    assert all(h.get("raw_bytes_read", 0) < 2_000_000 for h in report["hits"])


def test_merlin_camera_pixelsize_is_not_recommended_as_r(tmp_path: Path):
    from tools.emd_metadata import extract_emd_metadata

    path = tmp_path / "merlin.emd"
    with h5py.File(path, "w") as f:
        g = f.create_group("Data/Merlin")
        g.attrs["PixelSize"] = 0.055  # Å⁻¹/px-like; must not become nm/px
        g.attrs["Detector"] = "Merlin"
    report = extract_emd_metadata(path)
    r_hits = [h for h in report["hits"] if h.get("role") == "R" and h.get("nm_per_px")]
    assert not r_hits
    assert report["recommendation"] is None or report["recommendation"]["role"] != "R"


def test_scan_sources_compares_several_files(tmp_path: Path):
    from tools.emd_metadata import comparison_rows, scan_emd_sources

    a = _write_velox_emd(tmp_path / "a.emd", pixel_m=1.015e-10)
    sub = tmp_path / "nested"
    sub.mkdir()
    _write_velox_emd(sub / "b.emd", pixel_m=1.435e-10)
    reports = scan_emd_sources([tmp_path], recursive=True)
    names = {Path(r["path"]).name for r in reports}
    assert names == {"a.emd", "b.emd"}
    rows = comparison_rows(reports)
    assert len(rows) == 2
    by_name = {r["file"]: r for r in rows}
    assert by_name["a.emd"]["pm_per_px"] == pytest.approx(101.5, rel=1e-4)
    assert by_name["b.emd"]["pm_per_px"] == pytest.approx(143.5, rel=1e-3)


def test_filter_emd_paths_contains_and_exclude(tmp_path: Path):
    from tools.emd_metadata import filter_emd_paths

    names = [
        "0038 - STEM Micro HAADF-BF.emd",
        "0012 - Camera Micro Ceta.emd",
        "0039 - STEM Micro HAADF-BF.emd",
    ]
    paths = [tmp_path / n for n in names]
    kept = filter_emd_paths(paths, contains="HAADF", exclude="Ceta")
    assert [p.name for p in kept] == [
        "0038 - STEM Micro HAADF-BF.emd",
        "0039 - STEM Micro HAADF-BF.emd",
    ]
    only = filter_emd_paths(paths, contains=["0039", "0012"])
    assert [p.name for p in only] == [
        "0012 - Camera Micro Ceta.emd",
        "0039 - STEM Micro HAADF-BF.emd",
    ]
    assert filter_emd_paths(paths, contains="") == paths


def test_comparison_rows_include_image_resolution(tmp_path: Path):
    from tools.emd_metadata import comparison_rows, extract_emd_metadata

    path = tmp_path / "hires.emd"
    _write_velox_emd(path, pixel_m=7.176e-11)
    with h5py.File(path, "a") as f:
        del f["Data/Image/img0/Data"]
        f["Data/Image/img0"].create_dataset(
            "Data", data=np.zeros((4096, 4096, 1), dtype=np.uint16))
    report = extract_emd_metadata(path)
    assert report["image_shape"] == (4096, 4096)
    row = comparison_rows([report])[0]
    assert row["ny"] == 4096
    assert row["nx"] == 4096
    assert row["shape"] == "4096×4096"
    assert row["fov_nm"] == pytest.approx(4096 * 0.07176, rel=1e-3)
