"""Tests for orientation_peaks (Path A matcher + Path B smoke)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SI_CIF = Path(__file__).resolve().parent / "fixtures" / "Si.cif"


def test_match_theoretical_to_measured_units():
    import orientation_peaks as op

    Q = 0.02  # Å^-1 / px
    theo = [
        op.TheoreticalPeak(qx_A=0.4, qy_A=0.0, intensity=1.0, h=2, k=0, l=0),
        op.TheoreticalPeak(qx_A=0.0, qy_A=0.4, intensity=0.8, h=0, k=2, l=0),
        op.TheoreticalPeak(qx_A=0.0, qy_A=0.0, intensity=10.0, h=0, k=0, l=0),
    ]
    # Measured at exact theoretical positions (+ noise within tol)
    qx_px = np.array([0.4 / Q + 0.3, 0.0 / Q - 0.2, 5.0, 0.0])
    qy_px = np.array([0.0 / Q, 0.4 / Q + 0.1, 5.0, 0.0])
    inten = np.array([10.0, 8.0, 1.0, 100.0])
    matched = op.match_theoretical_to_measured(theo, qx_px, qy_px, inten, Q_pixel=Q, tol_px=1.0)
    assert len(matched) == 2
    hkls = {(m.h, m.k, m.l) for m in matched}
    assert (2, 0, 0) in hkls and (0, 2, 0) in hkls
    assert all(m.residual_px < 1.0 for m in matched)


def test_estimate_inplane_rotation_45deg():
    import orientation_peaks as op

    # Theoretical along +x; measured along +y ⇒ +90° rotation
    matched = [
        op.MatchedPeak(
            measured_index=0, qx_px=0, qy_px=20, qx_A=0.0, qy_A=0.4,
            intensity=1.0, h=2, k=0, l=0, residual_px=0.1, residual_A=0.002,
            theo_qx_A=0.4, theo_qy_A=0.0, theo_intensity=1.0,
        ),
        op.MatchedPeak(
            measured_index=1, qx_px=-20, qy_px=0, qx_A=-0.4, qy_A=0.0,
            intensity=0.8, h=0, k=2, l=0, residual_px=0.1, residual_A=0.002,
            theo_qx_A=0.0, theo_qy_A=0.4, theo_intensity=0.8,
        ),
    ]
    deg = op.estimate_inplane_rotation_deg(matched)
    assert deg is not None
    assert abs(deg - 90.0) < 1.0


def test_k_max_covering_bvm_expands_past_user():
    import orientation_peaks as op

    bvm = np.zeros((256, 256))
    origin = np.array([128.0, 128.0])
    Q = 0.01
    k_fov = op.k_max_covering_bvm(bvm, origin, Q)
    # Corner distance ≈ 128√2 px → ~1.81 Å⁻¹ (+ margin)
    assert k_fov > 1.8
    assert op.effective_generation_k_max(bvm, origin, Q, 0.5) == k_fov
    assert op.effective_generation_k_max(bvm, origin, Q, 3.0) == 3.0


def test_path_a_known_generate_synthetic_bvm():
    pytest.importorskip("py4DSTEM")
    pytest.importorskip("pymatgen")
    if not SI_CIF.is_file():
        pytest.skip("Si.cif fixture missing")

    from py4DSTEM.process.diffraction import Crystal
    import orientation_peaks as op

    crystal = Crystal.from_CIF(str(SI_CIF), primitive=False, conventional_standard_structure=True)
    Q = 0.01
    origin = np.array([64.0, 64.0])
    # Build a fake BVM and plant maxima at theoretical positions
    op.prepare_crystal_structure_factors(crystal, k_max=1.5)
    theo = op.generate_theoretical_peaks(
        crystal, zone_axis=(1, 1, 0), proj_x_lattice=(0, 0, -1), k_max=0.9
    )
    assert len(theo) >= 3

    bvm = np.zeros((128, 128), dtype=float)
    xs, ys, inten = [], [], []
    for t in theo[:12]:
        x = t.qx_A / Q + origin[0]
        y = t.qy_A / Q + origin[1]
        if 2 <= x < 126 and 2 <= y < 126:
            xi, yi = int(round(x)), int(round(y))
            bvm[yi, xi] = max(bvm[yi, xi], 200.0 + 100.0 * float(t.intensity))
            xs.append(x)
            ys.append(y)
            inten.append(200.0 + 100.0 * float(t.intensity))
    # Also paint origin
    bvm[int(origin[1]), int(origin[0])] = 500.0
    xs.append(origin[0])
    ys.append(origin[1])
    inten.append(500.0)

    maxima = {
        "x": np.asarray(xs, dtype=float),
        "y": np.asarray(ys, dtype=float),
        "intensity": np.asarray(inten, dtype=float),
    }
    result = op.run_known_generate(
        crystal=crystal,
        crystal_name="Si",
        bvm=bvm,
        origin_px=origin,
        Q_pixel=Q,
        zone_axis=(1, 1, 0),
        proj_x_lattice=(0, 0, -1),
        k_max=0.9,
        tol_px=2.0,
        maxima=maxima,
    )
    assert result.mode == "known_generate"
    assert result.n_matched >= 3
    assert result.rms_px < 2.0
    assert result.index_g1 != result.index_g2 or result.n_matched < 2


@pytest.mark.skipif(not SI_CIF.is_file(), reason="Si.cif fixture missing")
def test_path_b_acom_smoke():
    pytest.importorskip("py4DSTEM")
    pytest.importorskip("pymatgen")
    from py4DSTEM.process.diffraction import Crystal
    import orientation_peaks as op

    crystal = Crystal.from_CIF(str(SI_CIF), primitive=False, conventional_standard_structure=True)
    Q = 0.01
    origin = np.array([64.0, 64.0])
    op.prepare_crystal_structure_factors(crystal, k_max=1.2)
    theo = op.generate_theoretical_peaks(
        crystal, zone_axis=(1, 1, 0), proj_x_lattice=(0, 0, -1), k_max=0.9
    )
    xs, ys, inten = [origin[0]], [origin[1]], [500.0]
    bvm = np.zeros((128, 128), dtype=float)
    bvm[64, 64] = 500.0
    for t in theo[:10]:
        x = t.qx_A / Q + origin[0]
        y = t.qy_A / Q + origin[1]
        if 2 <= x < 126 and 2 <= y < 126:
            xs.append(x)
            ys.append(y)
            inten.append(100.0 + 50.0 * float(t.intensity))
            bvm[int(round(y)), int(round(x))] = inten[-1]
    maxima = {
        "x": np.asarray(xs, dtype=float),
        "y": np.asarray(ys, dtype=float),
        "intensity": np.asarray(inten, dtype=float),
    }
    result = op.run_acom_match(
        crystal=crystal,
        crystal_key=f"Si::{SI_CIF}",
        crystal_name="Si",
        bvm=bvm,
        origin_px=origin,
        Q_pixel=Q,
        k_max=0.9,
        tol_px=3.0,
        angle_step_zone_axis=8.0,
        angle_step_in_plane=8.0,
        maxima=maxima,
    )
    assert result.mode == "acom_match"
    assert result.orientation is not None
    assert result.n_theoretical >= 1
    # Correlation should be positive on self-consistent peaks
    assert result.corr_score is None or result.corr_score > 0.1
