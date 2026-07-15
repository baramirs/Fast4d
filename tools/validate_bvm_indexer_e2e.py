"""Headless E2E: index_bvm → apply indices → choose_basis_vectors ≈ manifest g1/g2.

Run with py4dstem-01419 from the peak-indexer-notebook worktree:
    python tools/validate_bvm_indexer_e2e.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import engine as E
import pipeline as pl

DEMO_BP = Path(r"C:\Users\jtapiaca.ASURITE\Desktop\Demo\256x256_Demo_braggpeaks.h5")
DEMO_MANIFEST = Path(
    r"C:\Users\jtapiaca.ASURITE\Desktop\Demo\256x256_Demo\data\strain_manifest.json"
)


def main() -> int:
    if not DEMO_BP.is_file() or not DEMO_MANIFEST.is_file():
        print("SKIP: Demo files missing")
        return 0

    manifest = json.loads(DEMO_MANIFEST.read_text(encoding="utf-8"))
    sbp = manifest["strain_basis_params"]
    cbv = sbp["choose_basis_vectors"]
    vis = cbv.get("vis_params", {})

    scan = E.Scan(
        name="256x256_Demo",
        braggpeaks_path=str(DEMO_BP),
    )
    p = scan.params
    p.center_guess = [float(v) for v in manifest["origin_xy"]]
    p.q_px = float(manifest["q_pixel_size"])
    p.q_refit = False
    p.qr_rotation = float(sbp.get("qr_rotation", 135.0))
    p.qr_flip = bool(sbp.get("qr_flip", False))
    p.min_spacing = int(cbv["minSpacing"])
    p.min_absolute_intensity = int(cbv["minAbsoluteIntensity"])
    p.max_num_peaks = int(cbv["maxNumPeaks"])
    p.edge_boundary = int(cbv["edgeBoundary"])
    p.vis_vmin = float(vis.get("vmin", 0.0))
    p.vis_vmax = float(vis.get("vmax", 0.995))
    p.max_peak_spacing = float(
        manifest["strain_params"]["set_max_peak_spacing"]["max_peak_spacing"]
    )
    p.zone_axis = [1, 1, 0]
    p.real_axis_h = [0, 0, -1]
    p.real_axis_v = [-1, 1, 0]
    p.cal_crystal = "Si"
    p.ellipse_enabled = False

    def log(msg: str) -> None:
        print(msg)

    print("=== index_bvm ===")
    result = E.index_bvm(scan, log=log, make_figure=False)
    print(
        f"propose origin={result.index_origin} g1={result.index_g1} g2={result.index_g2} "
        f"hkl={result.metrics.get('g1_hkl_str')}/{result.metrics.get('g2_hkl_str')}"
    )
    print(f"g1_px={result.g1_px}  g2_px={result.g2_px}")

    print("=== Send (apply indices) ===")
    E.apply_indexing_to_basis_params(scan, result, log=log)

    print("=== choose_basis_vectors with proposed indices ===")
    st = scan.ensure_state()
    pl.update_strain_basis_params(
        st,
        min_spacing=int(p.min_spacing),
        min_absolute_intensity=int(p.min_absolute_intensity),
        max_num_peaks=int(p.max_num_peaks),
        edge_boundary=int(p.edge_boundary),
        vmin=float(p.vis_vmin), vmax=float(p.vis_vmax),
        qr_rotation=float(p.qr_rotation), qr_flip=bool(p.qr_flip),
        manual_enabled=True,
        index_origin=int(p.index_origin),
        index_g1=int(p.index_g1),
        index_g2=int(p.index_g2),
        log=log,
    )
    strainmap = pl.setup_basis_step(st, log=log)
    g1 = np.asarray(strainmap.g1, dtype=float)
    g2 = np.asarray(strainmap.g2, dtype=float)
    g1_ref = np.asarray(sbp["g1_qxy"], dtype=float)
    g2_ref = np.asarray(sbp["g2_qxy"], dtype=float)

    print(f"g1 chosen: {g1}   manifest: {g1_ref}   Δ={g1 - g1_ref}")
    print(f"g2 chosen: {g2}   manifest: {g2_ref}   Δ={g2 - g2_ref}")

    # Allow sign flip / swap (indexing may propose -g or swapped axes)
    cands = [
        (g1, g2), (-g1, -g2), (g2, g1), (-g2, -g1),
        (g1, -g2), (-g1, g2), (-g2, g1), (g2, -g1),
    ]
    best = min(
        max(float(np.linalg.norm(a - g1_ref)), float(np.linalg.norm(b - g2_ref)))
        for a, b in cands
    )
    # Manifest indices were a specific manual pick; proposed indices should land on
    # the same lattice family. Strict 1e-3 only if indices match manifest exactly.
    man_idx = (int(cbv["index_origin"]), int(cbv["index_g1"]), int(cbv["index_g2"]))
    prop_idx = (int(p.index_origin), int(p.index_g1), int(p.index_g2))
    print(f"manifest indices {man_idx}  proposed {prop_idx}  best Δ={best:.6f} px")

    if prop_idx == man_idx:
        tol = 1e-3
        ok = best < tol
    else:
        # Same reciprocal family: vectors within ~1 px after sign/swap
        tol = 1.0
        ok = best < tol

    if not ok:
        print(f"FAIL: g1/g2 not within {tol} px of manifest (best={best})")
        return 1
    print(f"OK — g1/g2 within {tol} px of manifest family (best={best:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
