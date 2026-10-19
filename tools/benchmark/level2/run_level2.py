"""Level-2 error benchmark: drawn 4D patterns -> Fast4D detection -> calibration -> strain.
Protocol: docs/PROTOCOLO_BENCHMARK_ERRORES.md (nivel 2). Same guards as level 1
(py4DSTEM version + code hash) and the same provenance (env.json).

Usage:
    python tools/benchmark/level2/run_level2.py tools/benchmark/level2/<config>.json

Config: "reference", "output_dir", "workdir", "keep_raw", "seeds", "base" (all keys of
BASE_KEYS, explicit) and "sweeps" (list of {name, set, vary, values}), as in level 1.
Metrics per case (strained block = right half of the scan, reference = left half):
bias / sigma / RMSE of exx, eyy, exy against truth (py4DSTEM convention), total RMSE =
sqrt(mean over the 3 components of RMSE^2), rotation (theta) error against the drawn
lattice rotation, detection rate per disc family, position error, Q pixel, frame angle.
"""
from __future__ import annotations

import os

os.environ["TQDM_DISABLE"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import csv
import hashlib
import datetime as dt
import io
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parent)]
import benchmark_errors as B  # noqa: E402  (guards + provenance)
import gen4d as G  # noqa: E402

BASE_KEYS = ("n", "det", "q_px", "alpha_mrad", "dose", "strain_phys", "phi_deg",
             "phi_grad_deg", "qr_deg", "flip", "qr_deg_analysis", "flip_analysis",
             "frame_mode", "manual_rot_err_deg", "q_guess_factor", "detect", "n_vacuum",
             "vacuum_dose")
COMPS = ("exx", "eyy", "exy")


def _detection(bp, centers, n, tol=1.5) -> dict:
    pla = bp._v_uncal if hasattr(bp, "_v_uncal") else bp.vectors_uncal
    found = {f: [0, 0] for f in set(G.FAM)}
    errs, fp = [], 0
    for i in range(n):
        for j in range(n):
            pl = pla.get_pointlist(i, j).data
            q = np.column_stack([pl["qx"], pl["qy"]]) if len(pl) else np.zeros((0, 2))
            used = np.zeros(len(q), bool)
            for t, fam in zip(centers[i, j], G.FAM):
                found[fam][1] += 1
                if len(q):
                    d = np.hypot(*(q - t).T)
                    m = int(np.argmin(d))
                    if d[m] <= tol and not used[m]:
                        used[m] = True
                        found[fam][0] += 1
                        errs.append(q[m] - t)
            fp += int((~used).sum())
    e = np.array(errs) if errs else np.zeros((0, 2))
    return {"rate_" + f: found[f][0] / max(found[f][1], 1) for f in sorted(found)} | {
        "false_pos": fp,
        "pos_rms_px": float(np.sqrt(np.mean(np.sum(e ** 2, 1)))) if len(e) else float("nan")}


def run_case(case: dict, seed: int, work: Path, keep_raw: bool, log) -> dict:
    import driver as D
    import engine as E
    n, det = int(case["n"]), int(case["det"])
    name = f"c{seed}_{hashlib.sha1(json.dumps(case, sort_keys=True).encode()).hexdigest()[:10]}"
    folder = work / name
    truth = G.make_case(folder, name, n=n, det=det, q_px=float(case["q_px"]),
                        alpha_mrad=float(case["alpha_mrad"]), dose=case["dose"],
                        strain_phys=case["strain_phys"], phi_deg=float(case["phi_deg"]),
                        phi_grad_deg=float(case["phi_grad_deg"]), qr_deg=float(case["qr_deg"]),
                        flip=bool(case["flip"]), n_vacuum=int(case["n_vacuum"]),
                        vacuum_dose=float(case["vacuum_dose"]), seed=seed)
    sc = E.Scan(name=name, raw_path=str(folder / f"{name}.mib"), braggpeaks_path="")
    sc.vacuum_path = str(folder / f"{name}_vacuum.mib")
    p = sc.params
    p.center_guess = [det / 2, det / 2]
    p.q_px = float(case["q_px"]) * float(case["q_guess_factor"])
    p.roi_bounds = [0, n // 2, 0, n]
    p.cal_crystal, p.zone_axis = "Si", [1, 1, 0]
    p.frame_mode = str(case["frame_mode"])
    p.frame_x_hkl, p.frame_y_hkl = [-1, 1, 0], [0, 0, 1]
    # manual mode only: the frame angle the user types (true lattice angle + an error)
    p.coordinate_rotation = (float(case["phi_deg"]) + float(case["manual_rot_err_deg"])) % 360.0
    p.qr_rotation, p.qr_flip = float(case["qr_deg_analysis"]), bool(case["flip_analysis"])
    for k, v in case["detect"].items():
        if not hasattr(p, k):
            raise SystemExit(f"ABORT: unknown detection parameter {k!r}")
        setattr(p, k, v)
    opts = D.ComputeOptions(calibration="fit", do_without_roi=True, do_with_roi=True,
                            make_figures=False, figure_mode="off", save=False,
                            save_figures=False, save_braggpeaks=False)
    logs: list[str] = []
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        out = D.compute_scan(sc, opts, log=logs.append)
    for m in logs:
        log(f"[{name}] {m}")
    row = {"ok": bool(out.ok), "error": out.error or "", "elapsed_s": round(time.time() - t0, 1),
           "radius_px": truth["radius_px"], "discs_inside": truth["fraction_discs_inside"],
           "q_check": (p.q_px_check or {}).get("status", ""),
           "q_rel_err_pct": (float(p.q_px_fitted) / float(case["q_px"]) - 1) * 100
           if p.q_px_fitted else float("nan"),
           "n_warnings": sum("WARNING" in m for m in logs)}
    bp = getattr(sc.state, "braggpeaks", None)
    if bp is not None:
        row.update(_detection(bp, truth["centers"], n))
    if out.ok:
        raw = np.asarray(sc.state.strain_raw["with_roi"], float)   # eyy, exx, exy, theta, mask
        valid = raw[4] > 0.5
        exp = {"exx": truth["e_expected"][0, 0] * 100, "eyy": truth["e_expected"][1, 1] * 100,
               "exy": truth["e_expected"][0, 1] * 100}
        rec = {"eyy": raw[0] * 100, "exx": raw[1] * 100, "exy": raw[2] * 100}
        strd, ref = np.s_[:, n // 2:], np.s_[:, : n // 2]
        tot = []
        for c in COMPS:
            a = np.where(valid, rec[c], np.nan)
            s_ = a[strd][np.isfinite(a[strd])]
            r_ = a[ref][np.isfinite(a[ref])]
            row[f"{c}_bias"] = float(s_.mean() - exp[c])
            row[f"{c}_sigma"] = float(s_.std())
            row[f"{c}_rmse"] = float(np.sqrt(np.mean((s_ - exp[c]) ** 2)))
            row[f"{c}_ref_rmse"] = float(np.sqrt(np.mean(r_ ** 2)))
            tot.append(row[f"{c}_rmse"] ** 2)
        row["rmse_total"] = float(np.sqrt(np.mean(tot)))
        # py4DSTEM theta has the opposite sense to the drawn CCW lattice rotation (pilot
        # 2026-10-19, phi_grad 2 deg: RMSE 0.07 deg with the minus sign, 1.25 without)
        th = np.degrees(raw[3])
        th_true = -(truth["phi_map"] - float(np.median(truth["phi_map"][ref])))
        ok_t = valid & np.isfinite(th)
        row["theta_bias_deg"] = float(np.mean((th - th_true)[ok_t]))
        row["theta_rmse_deg"] = float(np.sqrt(np.mean((th - th_true)[ok_t] ** 2)))
        row["valid_frac"] = float(valid.mean())
        card = (sc.state.crystal_frame_cards or {}).get("with_roi", {})
        ang = card.get("x_angle_applied_deg")
        row["frame_err_deg"] = (float(B.wrap_axial_deg(float(ang) - float(case["phi_deg"])))
                                if ang is not None else float("nan"))
    # release the datacube / memory maps before deleting the files: an open memmap keeps the
    # disk space allocated (a 216-run stage filled the disk this way, 2026-10-19)
    E.release_scans([sc], log=lambda m: None)
    # the vacuum probe cache keeps the vacuum .mib memmap open (and writes .npz copies):
    # each case has its own vacuum file, so forget it here
    E._pipeline().clear_vacuum_probe_cache(disk=True)
    del sc, bp, out
    import gc
    gc.collect()
    if not keep_raw:
        shutil.rmtree(folder)
    return row


def main(cfg_path: str) -> None:
    cfg = json.loads(Path(cfg_path).read_text())
    p4 = B.check_py4dstem(cfg["reference"])
    env = B.environment(p4)
    cases = B.build_cases_generic(cfg, BASE_KEYS)
    seeds = [int(s) for s in cfg["seeds"]]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / cfg["output_dir"] / f"{stamp}_{env['fast4d_commit'][:7]}_{platform.system()}_L2"
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "env.json").write_text(json.dumps(env, indent=2))
    work = Path(cfg["workdir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    logf = (out / "run.log").open("w")
    log = lambda m: logf.write(m + "\n")  # noqa: E731
    rows, t0 = [], time.time()
    # per-run cache so an interrupted run (session restart) resumes; the key includes the
    # py4DSTEM and Fast4D code hashes, so a code change never reuses old results
    cache = work / "cache"
    cache.mkdir(exist_ok=True)
    # cache key: Fast4D core (repo-root *.py) + the data generator; a change in the metric
    # code of this script needs an empty cache (delete <workdir>/cache)
    hcore = hashlib.sha256()
    for f in sorted(REPO.glob("*.py")) + [HERE / "gen4d.py"]:
        hcore.update(f.name.encode()); hcore.update(f.read_bytes())
    core_hash = hcore.hexdigest()
    for factor, value, case in cases:
        for seed in seeds:
            key = hashlib.sha1(json.dumps([case, seed, p4["sha256"], core_hash],
                                          sort_keys=True).encode()).hexdigest()[:16]
            cf = cache / f"{key}.json"
            if cf.is_file():
                r = json.loads(cf.read_text())
            else:
                stale = work / f"c{seed}_{hashlib.sha1(json.dumps(case, sort_keys=True).encode()).hexdigest()[:10]}"
                if stale.exists():
                    shutil.rmtree(stale)              # half-written data from an interruption
                r = run_case(case, seed, work, bool(cfg["keep_raw"]), log)
                cf.write_text(json.dumps(r))
            rows.append({"factor": factor, "value": json.dumps(value), "seed": seed, **r})
            logf.flush()
            print(f"  {factor}={json.dumps(value)} seed {seed}: ok={r['ok']} "
                  f"rmse={r.get('rmse_total', float('nan')):.4f} ({r['elapsed_s']} s)", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("factor", "value", "seed"), k))
    with open(out / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(
        {"env": {k: env[k] for k in env if k != "packages"}, "elapsed_s": round(time.time() - t0, 1),
         "rows": rows}, indent=1, default=str))
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/benchmark/level2/run_level2.py <config.json>")
    main(sys.argv[1])
