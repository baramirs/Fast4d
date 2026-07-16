"""Tests for Crystal-from-CIF (Index BVM + shared Q-pixel via _build_crystal)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE_CIF = Path(__file__).resolve().parent / "fixtures" / "Si.cif"


@pytest.fixture(scope="module")
def si_cif_path() -> Path:
    if not FIXTURE_CIF.is_file():
        pytest.skip(f"fixture CIF missing: {FIXTURE_CIF}")
    return FIXTURE_CIF


class TestLoadCrystalFromCif:
    def test_si_fixture_loads_cubic(self, si_cif_path: Path):
        import engine as E

        info = E.load_crystal_from_cif(si_cif_path)
        assert info.is_cubic is True
        assert info.warning is None
        assert abs(info.cal.a_lat - 5.431) < 1e-3
        assert len(info.cal.positions) >= 2
        assert Path(info.path).resolve() == si_cif_path.resolve()

    def test_missing_path_raises(self, tmp_path: Path):
        import engine as E

        with pytest.raises(FileNotFoundError):
            E.load_crystal_from_cif(tmp_path / "nope.cif")

    def test_non_cubic_warns(self, tmp_path: Path):
        """Hexagonal CIF → is_cubic False + warning (v1 still returns effective a)."""
        pytest.importorskip("pymatgen")
        from pymatgen.core import Lattice, Structure

        import engine as E

        hex_s = Structure(
            Lattice.hexagonal(3.0, 5.0),
            ["C", "C"],
            [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5]],
        )
        path = tmp_path / "hex.cif"
        hex_s.to(filename=str(path), fmt="cif")
        info = E.load_crystal_from_cif(path)
        assert info.is_cubic is False
        assert info.warning
        assert "not cubic" in info.warning.lower()
        assert info.cal.a_lat > 0


class TestCalCrystalObjCif:
    def test_cal_crystal_obj_reads_cif_path(self, si_cif_path: Path):
        import engine as E

        p = E.CalibrationParams(cal_crystal="CIF", cif_path=str(si_cif_path))
        cc = p.cal_crystal_obj()
        assert abs(cc.a_lat - 5.431) < 1e-3
        assert cc.name  # stem / formula

    def test_cif_without_path_falls_back_to_default(self):
        import engine as E

        p = E.CalibrationParams(cal_crystal="CIF", cif_path=None)
        cc = p.cal_crystal_obj()
        assert cc.name == E.DEFAULT_CAL_CRYSTAL
        assert abs(cc.a_lat - E.CAL_CRYSTALS["Si"].a_lat) < 1e-9


class TestBuildCrystalCif:
    def test_build_crystal_from_cif(self, si_cif_path: Path):
        import engine as E

        class FakeScan:
            params = E.CalibrationParams(cal_crystal="CIF", cif_path=str(si_cif_path))

        crystal = E._build_crystal(FakeScan())
        assert crystal is not None
        cell = np.asarray(crystal.cell, dtype=float).ravel()
        assert abs(float(cell[0]) - 5.431) < 1e-3
        assert len(crystal.positions) >= 2

    def test_to_dict_persists_cif_path(self, si_cif_path: Path):
        import engine as E

        p = E.CalibrationParams(cal_crystal="CIF", cif_path=str(si_cif_path))
        d = p.to_dict()
        assert d["cal_crystal"] == "CIF"
        assert d["cif_path"] == str(si_cif_path)

        p2 = E.CalibrationParams()
        E._overlay_params_dict(p2, d)
        assert p2.cal_crystal == "CIF"
        assert p2.cif_path == str(si_cif_path)


class TestIsApproximatelyCubic:
    def test_cubic_cell(self):
        import engine as E

        assert E.is_approximately_cubic((5.431, 5.431, 5.431, 90.0, 90.0, 90.0))

    def test_hex_cell(self):
        import engine as E

        assert not E.is_approximately_cubic((3.0, 3.0, 5.0, 90.0, 90.0, 120.0))
