"""Tests for bvm_indexing motor (RANSAC + hkl anchor + propose indices)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import bvm_indexing as bvm


def _synthetic_lattice(
    a: np.ndarray,
    b: np.ndarray,
    *,
    indices: list[tuple[int, int]],
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build points = m*a + n*b (+ optional noise). Returns points, weights."""
    rng = rng or np.random.default_rng(0)
    pts = []
    for m, n in indices:
        p = m * a + n * b
        if noise:
            p = p + rng.normal(0.0, noise, size=2)
        pts.append(p)
    points = np.asarray(pts, dtype=float)
    weights = np.linspace(1.0, 0.3, len(points))
    return points, weights


class TestRansac:
    def test_recovers_basis_up_to_unimodular(self):
        a = np.array([0.32, 0.05])
        b = np.array([0.08, 0.30])
        idxs = [(m, n) for m in range(-3, 4) for n in range(-3, 4) if (m, n) != (0, 0)]
        # Add a few outliers
        points, weights = _synthetic_lattice(a, b, indices=idxs, noise=0.002)
        outliers = np.array([[0.9, 0.1], [0.1, 0.85], [-0.7, 0.6]])
        points = np.vstack([points, outliers])
        weights = np.concatenate([weights, np.full(3, 0.05)])

        lat = bvm.fit_lattice_ransac(points, weights, tol_A=0.03, seed=1, n_iterations=2000)
        B = np.column_stack([lat["vector_a"], lat["vector_b"]])
        M = np.linalg.solve(B, np.column_stack([a, b]))
        M_int = np.round(M)
        assert np.max(np.abs(M - M_int)) < 0.08
        assert abs(int(round(np.linalg.det(M_int)))) == 1

    def test_score_rejects_singular(self):
        pts = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        w = np.ones(3)
        score, inl, _ = bvm.score_basis(pts, w, np.array([1.0, 0.0]), np.array([2.0, 0.0]), 0.1)
        assert score == -np.inf or not inl.any()


class TestZoneAndHkl:
    def test_zolz_weiss_law(self):
        zolz = bvm.enumerate_zolz([1, 1, 0], max_index=3)
        assert len(zolz) > 0
        assert np.all(zolz @ np.array([1, 1, 0]) == 0)
        assert not any(tuple(r) == (0, 0, 0) for r in zolz)

    def test_match_g1g2_si110(self):
        # Si [110]: g1 ~ (2-20), g2 ~ (002) with a=5.4309
        a_lat = 5.4309
        G1 = np.array([-2, 2, 0], dtype=float)
        G2 = np.array([0, 0, -2], dtype=float)
        # Put them at ~90° in detector space as Å⁻¹ magnitudes
        g1_A = np.array([np.linalg.norm(G1) / a_lat, 0.0])
        g2_A = np.array([0.0, np.linalg.norm(G2) / a_lat])
        H1, H2, cost = bvm.match_g1g2_hkl(g1_A, g2_A, [1, 1, 0], a_lat)
        assert abs(np.linalg.norm(H1) - np.linalg.norm(G1)) < 1e-9 or np.linalg.norm(H1) == np.linalg.norm(G1)
        assert cost < 1.0
        assert int(np.dot(H1, [1, 1, 0])) == 0
        assert int(np.dot(H2, [1, 1, 0])) == 0

    def test_anchor_signs(self):
        # Synthetic: QR=135°, real axes map as in the notebook
        a_lat = 5.4309
        # g1 along vertical-ish after rotation; use known Demo-like vectors
        g1 = np.array([-26.88, 26.86])  # px
        g2 = np.array([-18.54, -18.88])
        Q = 0.01382376
        mag1 = float(np.hypot(*(g1 * Q)))
        mag2 = float(np.hypot(*(g2 * Q)))
        G1, G2, s = bvm.anchor_hkl_with_real_axes(
            g1, g2, mag1, mag2,
            zone_axis=[1, 1, 0],
            real_axis_h=[0, 0, -1],
            real_axis_v=[-1, 1, 0],
            qr_rotation_deg=135.0,
            lattice_a=a_lat,
        )
        assert int(np.dot(G1, [1, 1, 0])) == 0
        assert int(np.dot(G2, [1, 1, 0])) == 0
        # Expect ±(2-20) / ±(002) family (orders may vary)
        n1 = float(np.linalg.norm(G1))
        n2 = float(np.linalg.norm(G2))
        assert abs(n1 - 2 * np.sqrt(2)) < 0.1 or abs(n1 - 2.0) < 0.1
        assert abs(n2 - 2.0) < 0.1 or abs(n2 - 2 * np.sqrt(2)) < 0.1
        assert s in (+1, -1)


class TestProposeIndices:
    def test_propose_basis_indices(self):
        # Synthetic maxima relative to origin
        qx = np.array([0.0, 10.0, 0.0, 5.0, -10.0])
        qy = np.array([0.0, 0.0, 12.0, 5.0, 0.0])
        i0, i1, i2 = bvm.propose_basis_indices(
            qx, qy, np.array([10.0, 0.0]), np.array([0.0, 12.0])
        )
        assert i0 == 0
        assert i1 == 1
        assert i2 == 2


class TestIndexBvmSynthetic:
    def test_end_to_end_synthetic_bvm(self):
        # Delta peaks on a known lattice → get_maxima_2D + RANSAC recover integer relation
        N = 128
        origin = np.array([64.0, 64.0])
        a_px = np.array([12.0, 4.0])
        b_px = np.array([-3.0, 11.0])
        bvm_img = np.zeros((N, N), dtype=float)
        idxs = [(m, n) for m in range(-3, 4) for n in range(-3, 4)]
        for m, n in idxs:
            cx = int(round(origin[0] + m * a_px[0] + n * b_px[0]))
            cy = int(round(origin[1] + m * a_px[1] + n * b_px[1]))
            if 2 <= cx < N - 2 and 2 <= cy < N - 2:
                bvm_img[cx, cy] = 500.0 / (1.0 + 0.15 * (abs(m) + abs(n)))

        Q = 0.02
        maxima = bvm.find_bvm_maxima(
            bvm_img,
            min_spacing=5,
            min_absolute_intensity=20,
            max_num_peaks=40,
            edge_boundary=2,
            subpixel="pixel",
        )
        assert len(maxima) >= 8
        qx = maxima["x"] - origin[0]
        qy = maxima["y"] - origin[1]
        points = np.column_stack([qx * Q, qy * Q])
        lat = bvm.fit_lattice_ransac(
            points, weights=maxima["intensity"], tol_A=2.5 * Q, seed=0, n_iterations=2000
        )
        B = np.column_stack([lat["vector_a"] / Q, lat["vector_b"] / Q])
        M = np.linalg.solve(B, np.column_stack([a_px, b_px]))
        assert np.max(np.abs(M - np.round(M))) < 0.15
        assert abs(int(round(np.linalg.det(np.round(M))))) >= 1
        assert int(np.sum(lat["inliers"])) >= 8


DEMO_BP = Path(r"C:\Users\jtapiaca.ASURITE\Desktop\Demo\256x256_Demo_braggpeaks.h5")
DEMO_MANIFEST = Path(
    r"C:\Users\jtapiaca.ASURITE\Desktop\Demo\256x256_Demo\data\strain_manifest.json"
)


@pytest.mark.skipif(not DEMO_BP.is_file() or not DEMO_MANIFEST.is_file(),
                    reason="Demo braggpeaks/manifest not present")
class TestDemoRegression:
    def test_index_bvm_demo_proposes_near_manifest(self):
        import json
        import sys
        from pathlib import Path as P

        root = P(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import pipeline as pl
        from state import WorkflowState

        manifest = json.loads(DEMO_MANIFEST.read_text(encoding="utf-8"))
        bp = pl.load_braggpeaks_file(
            DEMO_BP, expected_rshape=tuple(manifest["image_shape"]), log=None
        )
        state = WorkflowState()
        state.braggpeaks = bp
        state.braggpeaks_path = DEMO_BP
        oy, ox = (int(v) for v in manifest["origin_xy"])
        pl.set_origin_center_guess(state, (oy, ox), sampling=1, log=None)
        pl.run_origin_correction_step(state, log=None)
        pl.set_q_pixel_size_step(
            state, float(manifest["q_pixel_size"]),
            units=str(manifest["q_pixel_units"]), log=None,
        )
        sbp = manifest["strain_basis_params"]
        cbv = sbp["choose_basis_vectors"]
        vis = cbv.get("vis_params", {})
        pl.update_strain_basis_params(
            state,
            min_spacing=int(cbv["minSpacing"]),
            min_absolute_intensity=int(cbv["minAbsoluteIntensity"]),
            max_num_peaks=int(cbv["maxNumPeaks"]),
            edge_boundary=int(cbv["edgeBoundary"]),
            vmin=float(vis.get("vmin", 0.0)),
            vmax=float(vis.get("vmax", 0.995)),
            qr_rotation=float(sbp.get("qr_rotation", 0.0)),
            qr_flip=bool(sbp.get("qr_flip", False)),
            manual_enabled=False,
            log=None,
        )

        bvm_cal = np.asarray(bp.histogram(mode="cal", sampling=1).data, dtype=float)
        origin = np.asarray(bp.calibration.get_origin_mean(), dtype=float)
        Q = float(bp.calibration.get_Q_pixel_size())
        result = bvm.index_bvm(
            bvm_cal, origin,
            Q_pixel=Q,
            Q_units=str(bp.calibration.get_Q_pixel_units()),
            lattice_a=5.4309,
            zone_axis=[1, 1, 0],
            real_axis_h=[0, 0, -1],
            real_axis_v=[-1, 1, 0],
            qr_rotation_deg=float(sbp.get("qr_rotation", 135.0)),
            tol_px=float(manifest["strain_params"]["set_max_peak_spacing"]["max_peak_spacing"]),
            seed=0,
            min_spacing=int(cbv["minSpacing"]),
            min_absolute_intensity=int(cbv["minAbsoluteIntensity"]),
            max_num_peaks=int(cbv["maxNumPeaks"]),
            edge_boundary=int(cbv["edgeBoundary"]),
        )

        g1_ref = np.asarray(sbp["g1_qxy"], dtype=float)
        g2_ref = np.asarray(sbp["g2_qxy"], dtype=float)
        # Proposed g vectors should match manifest within a few px (same lattice family)
        # Allow sign flip / swap of g1↔g2
        cands = [
            (result.g1_px, result.g2_px),
            (-result.g1_px, -result.g2_px),
            (result.g2_px, result.g1_px),
            (-result.g2_px, -result.g1_px),
            (result.g1_px, -result.g2_px),
            (-result.g1_px, result.g2_px),
        ]
        best = min(
            max(float(np.linalg.norm(c1 - g1_ref)), float(np.linalg.norm(c2 - g2_ref)))
            for c1, c2 in cands
        )
        assert best < 1.5, f"proposed g1/g2 far from manifest (best Δ={best:.3f} px)"
        assert result.n_inliers >= 20
        assert int(np.dot(result.g1_hkl, [1, 1, 0])) == 0
        assert int(np.dot(result.g2_hkl, [1, 1, 0])) == 0
