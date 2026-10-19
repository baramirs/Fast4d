"""Export the level-1 synthetic datasets (with their ground truth) so they can be opened in
the Fast4D GUI or any py4DSTEM script. Protocol: docs/PROTOCOLO_BENCHMARK_ERRORES.md

Usage:
    python tools/benchmark/export_synthetic.py tools/benchmark/config_level1.json <out_dir>

For each exported case <out_dir>/<case>/ contains:
  braggpeaks.h5   py4DSTEM BraggVectors (raw, uncalibrated), loadable with
                  engine.load_braggpeaks / the GUI "braggpeaks" file
  truth.npz       eyy_phys (physical strain, fraction), eyy_expected_global (what py4DSTEM
                  must read against the median of the whole crystal), qx0, qy0 (beam origin,
                  px), vacuum (bool), exx/exy truth = 0
  case.json       generation parameters + Fast4D parameters needed to reproduce the analysis
Every case is re-analysed after writing (load from disk -> calibration -> strain) and the
error is checked against the in-memory benchmark result: the export fails loudly otherwise.
"""
from __future__ import annotations

import os

os.environ["TQDM_DISABLE"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_errors as B  # noqa: E402  (same guards, same case runner)

# name -> overrides of config["base"]; seed is explicit per case
CASES = {
    "01_base": {},
    "02_no_noise": {"jitter_px": 0.0},
    "03_unstrained": {"e_max": 0.0},
    "04_strong_10pct": {"e_max": 0.10},
    "05_ellipse_1pct_uncorrected": {"ellipse": [1.01, 1.0, 25.0]},
    "06_vacuum_band_roi": {"vacuum_cols": 6, "roi": [2, 12, 8, 16]},
}
SEED = 0


def maps_figure(path: Path, name: str, eyy, truth_eyy, valid) -> None:
    """Truth / Fast4D / error maps of e_yy (% strain), diverging blue-grey-red."""
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    cmap.set_bad("#c3c2b7")
    m = lambda a: np.where(valid, a * 100.0, np.nan)
    t, f = m(truth_eyy), m(eyy)
    err = f - t
    lim = float(np.nanmax(np.abs(np.r_[t.ravel(), f.ravel()]))) or 1e-3
    elim = float(np.nanmax(np.abs(err))) or 1e-6
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), facecolor="#fcfcfb")
    for ax, a, title, l in ((axs[0], t, "truth e_yy", lim), (axs[1], f, "Fast4D e_yy", lim),
                             (axs[2], err, "error (Fast4D - truth)", elim)):
        im = ax.imshow(a, cmap=cmap, vmin=-l, vmax=l, interpolation="nearest")
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("% strain", fontsize=8, color="#52514e")
        cb.ax.tick_params(labelsize=7, colors="#898781", labelcolor="#52514e")
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    fig.suptitle(f"{name}: RMSE {rmse:.4f} % (grey = no lattice fit / vacuum)", fontsize=10,
                 x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(cfg_path: str, out_dir: str) -> None:
    import py4DSTEM
    import engine as E
    import synthetic_si110 as S

    cfg = json.loads(Path(cfg_path).read_text())
    p4 = B.check_py4dstem(cfg["reference"])
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, over in CASES.items():
        case = dict(cfg["base"]); case.update(over)
        n = int(case["n"])
        vac = S.vacuum_columns(n, int(case["vacuum_cols"])) if case["vacuum_cols"] else None
        ell = tuple(case["ellipse"]) if case["ellipse"] is not None else None
        bv, truth = S.make_braggvectors(n, e_max=float(case["e_max"]), q_pix=S.Q_PIX,
                                        phi_deg=S.PHI_DEG, g_max=float(case["g_max"]),
                                        jitter_px=float(case["jitter_px"]), vacuum=vac,
                                        ellipse=ell, seed=SEED)
        d = out / name
        d.mkdir()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            py4DSTEM.save(str(d / "braggpeaks.h5"), bv, mode="o")
        np.savez_compressed(d / "truth.npz", eyy_phys=truth["eyy_phys"],
                            eyy_expected_global=S.expected_eyy(truth), qx0=truth["qx0"],
                            qy0=truth["qy0"], vacuum=truth["vacuum"])
        info = {
            "case": name, "seed": SEED, "generation": case,
            "truth": {"crystal": "Si diamond a=5.431 A", "zone_axis": [1, 1, 0],
                      "q_pixel_1_per_A": S.Q_PIX, "x_axis_[-110]_angle_deg": S.PHI_DEG,
                      "strain": "physical e_yy only (along [001]), linear ramp, 0 at centre",
                      "pattern_shape": list(S.QSHAPE), "scan_shape": [n, n]},
            "fast4d_params": {"cal_crystal": "Si", "zone_axis": [1, 1, 0],
                              "center_guess": [128.0 + case["center_err_px"]] * 2,
                              "q_px": case["q_guess"], "q_kmax": 1.6,
                              "frame_mode": case["frame_mode"], "frame_x_hkl": [-1, 1, 0],
                              "frame_y_hkl": [0, 0, 1], "roi_bounds": case["roi"],
                              "ellipse_enabled": False},
            "py4dstem": p4,
        }
        (d / "case.json").write_text(json.dumps(info, indent=2))

        # round trip: the file on disk must give the same result as the in-memory run
        mem = B.run_case(case, SEED, log=lambda m: None)
        sc = S.make_scan(None, name=name, roi_bounds=case["roi"],
                         frame_mode=str(case["frame_mode"]),
                         coordinate_rotation=S.PHI_DEG + float(case["rot_err_deg"]),
                         q_guess=float(case["q_guess"]))
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            E.load_braggpeaks(sc, str(d / "braggpeaks.h5"), log=lambda m: None)
            sc.params.center_guess = [128.0 + case["center_err_px"]] * 2
            sc.params.indexing_seed = SEED
            sc.params.ellipse_enabled = False
            E.run_calibration_sequence(sc, make_figures=False, log=lambda m: None)
            E.compute_strain(sc, use_roi=False, log=lambda m: None)
            if case["roi"] is not None:
                E.compute_strain(sc, use_roi=True, log=lambda m: None)
        label = "with_roi" if case["roi"] is not None else "without_roi"
        raw = np.asarray(sc.state.strain_raw[label], float)
        v = raw[4] > 0.5
        ref = E._reference_roi_mask(sc, sc.state.strainmap_full.rshape) if case["roi"] else None
        rmse = float(np.sqrt(np.mean(((raw[0] - S.expected_eyy(truth, ref))[v] * 100) ** 2)))
        if abs(rmse - mem["rmse_eyy_pct"]) > 1e-10:
            raise RuntimeError(f"{name}: file round trip RMSE {rmse} != in-memory {mem['rmse_eyy_pct']}")
        np.savez_compressed(d / "fast4d_measured.npz", eyy=raw[0], exx=raw[1], exy=raw[2],
                            valid=v, eyy_expected=S.expected_eyy(truth, ref), map_label=label)
        maps_figure(d / "maps.png", name, raw[0], S.expected_eyy(truth, ref), v)
        rows.append((name, mem["rmse_eyy_pct"], mem["rms_exy_pct"], mem["q_px_rel_err_pct"]))

    print(f"py4DSTEM {p4['version']} sha {p4['sha256'][:12]} -> {out}")
    print(f"{'case':>30} {'RMSE eyy%':>10} {'exy%':>7} {'dq%':>8}  (file round trip OK)")
    for r in rows:
        print(f"{r[0]:>30} {r[1]:10.4f} {r[2]:7.3f} {r[3]:8.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/benchmark/export_synthetic.py <config.json> <out_dir>")
    main(sys.argv[1], sys.argv[2])
