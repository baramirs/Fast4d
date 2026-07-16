"""One-off generator for tests/fixtures/Si.cif — NOT a test, run manually.

Builds the diamond-cubic Si structure (a=5.431 A, Fd-3m) with pymatgen and
writes a CIF, so the fixture used by test_crystal_cif.py has no network
dependency and matches the values already hardcoded in engine.CAL_CRYSTALS.
"""
from __future__ import annotations

from pathlib import Path

from pymatgen.core import Lattice, Structure

A_SI = 5.431

if __name__ == "__main__":
    lattice = Lattice.cubic(A_SI)
    # Diamond cubic basis (8 sites / conventional cell), matches engine._DIAMOND_POS.
    coords = [
        [0.0, 0.0, 0.0], [0.25, 0.25, 0.25],
        [0.0, 0.5, 0.5], [0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5], [0.75, 0.25, 0.75],
        [0.5, 0.5, 0.0], [0.75, 0.75, 0.25],
    ]
    structure = Structure(lattice, ["Si"] * len(coords), coords)
    out = Path(__file__).parent / "Si.cif"
    structure.to(filename=str(out), fmt="cif")
    print(f"wrote {out}")
