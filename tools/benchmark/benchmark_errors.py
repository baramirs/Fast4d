"""Fast4D error benchmark against synthetic ground truth (level 1).

Protocol: docs/PROTOCOLO_BENCHMARK_ERRORES.md

Level 1 = Bragg peaks injected straight into a py4DSTEM BraggVectors
(tests/synthetic_si110.py). It measures the error of calibration + lattice fit +
strain computation in Fast4D's own engine (the functions the GUI calls). It does NOT
include Bragg disk detection error.

Usage (official run on Windows: run_benchmark.bat):
    python tools/benchmark/benchmark_errors.py tools/benchmark/config_level1.json

Guards (the run aborts, no results are written):
  * py4DSTEM.__version__ must equal config["reference"]["py4dstem_version"]
  * SHA-256 of the installed py4DSTEM *.py files must equal
    config["reference"]["py4dstem_sha256"] (same code as the GUI environment)

Every config parameter is required: there are no defaults in this script.
"""
from __future__ import annotations

import os

os.environ["TQDM_DISABLE"] = "1"            # progress bars off: short console output
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import logging
import platform
import sys
import time
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

BASE_KEYS = ("n", "e_max", "jitter_px", "g_max", "q_guess", "frame_mode", "rot_err_deg",
             "center_err_px", "ellipse", "vacuum_cols", "roi")
METRICS = ("bias_eyy_pct", "rmse_eyy_pct", "maxabs_eyy_pct", "p95_eyy_pct",
           "rms_exx_pct", "rms_exy_pct", "q_px_rel_err_pct", "angle_err_deg",
           "fit_fail_frac")


# ── provenance ────────────────────────────────────────────────────────────────

def sha256_of_py_files(root: Path) -> str:
    """Hash of every *.py under ``root`` (sorted relative POSIX paths + contents).
    Same recipe as the reference hash taken from the GUI environment:
    ``find py4DSTEM -name '*.py' | sort | xargs sha256sum | sha256sum``."""
    lines = []
    for f in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root.parent).as_posix()):
        rel = f.relative_to(root.parent).as_posix()
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def fast4d_commit() -> str:
    """HEAD commit read from .git without needing the git binary."""
    git = REPO / ".git"
    if git.is_file():                                   # worktree: "gitdir: <path>"
        git = Path(git.read_text().split(":", 1)[1].strip())
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(" ", 1)[1]
    loose = git / ref
    if loose.is_file():
        return loose.read_text().strip()
    for line in (git / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line.split(" ", 1)[0]
    raise RuntimeError(f"cannot resolve git ref {ref}")


def fast4d_code_sha256() -> str:
    """Hash of Fast4D's own *.py (repo root + tests + tools), covers uncommitted edits."""
    files = sorted([*REPO.glob("*.py"), *(REPO / "tests").glob("*.py"),
                    *(REPO / "tools").rglob("*.py")], key=lambda p: p.relative_to(REPO).as_posix())
    h = hashlib.sha256()
    for f in files:
        if "__pycache__" in f.parts:
            continue
        h.update(f.relative_to(REPO).as_posix().encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def check_py4dstem(ref: dict) -> dict:
    import py4DSTEM
    ver = py4DSTEM.__version__
    if ver != ref["py4dstem_version"]:
        raise SystemExit(f"ABORT: py4DSTEM {ver} != required {ref['py4dstem_version']} "
                         f"({py4DSTEM.__file__}). Use the GUI environment.")
    h = sha256_of_py_files(Path(py4DSTEM.__file__).parent)
    if h != ref["py4dstem_sha256"]:
        raise SystemExit(f"ABORT: py4DSTEM code hash {h} != reference {ref['py4dstem_sha256']}. "
                         "The installed py4DSTEM is not the GUI's copy.")
    return {"version": ver, "sha256": h, "path": str(Path(py4DSTEM.__file__).parent)}


def environment(p4: dict) -> dict:
    pkgs = sorted({(d.metadata["Name"] or "").lower(): d.version
                   for d in metadata.distributions() if d.metadata["Name"]}.items())
    import numpy, scipy
    return {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "py4dstem": p4, "numpy": numpy.__version__, "scipy": scipy.__version__,
        "fast4d_commit": fast4d_commit(), "fast4d_code_sha256": fast4d_code_sha256(),
        "packages": dict(pkgs),
    }


# ── one synthetic case ────────────────────────────────────────────────────────

def wrap_axial_deg(a: float) -> float:
    """Angle difference modulo 180 deg (x and -x give the same strain tensor), in (-90, 90]."""
    return float((a + 90.0) % 180.0 - 90.0)


def run_case(case: dict, seed: int, log) -> dict:
    import engine as E
    import synthetic_si110 as S

    n = int(case["n"])
    vac = S.vacuum_columns(n, int(case["vacuum_cols"])) if case["vacuum_cols"] else None
    ellipse = tuple(case["ellipse"]) if case["ellipse"] is not None else None
    bv, truth = S.make_braggvectors(n, e_max=float(case["e_max"]), q_pix=S.Q_PIX,
                                    phi_deg=S.PHI_DEG, g_max=float(case["g_max"]),
                                    jitter_px=float(case["jitter_px"]), vacuum=vac,
                                    ellipse=ellipse, seed=int(seed))
    sc = S.make_scan(bv, name=f"bench_s{seed}", roi_bounds=case["roi"],
                     frame_mode=str(case["frame_mode"]),
                     coordinate_rotation=S.PHI_DEG + float(case["rot_err_deg"]),
                     q_guess=float(case["q_guess"]))
    dc = float(case["center_err_px"])
    sc.params.center_guess = [128.0 + dc, 128.0 + dc]
    sc.params.indexing_seed = int(seed)
    sc.params.ellipse_enabled = False     # level 1: an ellipse in the data stays uncorrected

    E.run_calibration_sequence(sc, make_figures=False, log=log)
    use_roi = case["roi"] is not None
    # Same order as driver.py (GUI): global map first, then the ROI-referenced map.
    # (with-ROI alone after calibration fails: strainmap_full from the basis step has
    # no g1g2_map yet; see protocol, "hallazgos".)
    E.compute_strain(sc, use_roi=False, log=log)
    if use_roi:
        E.compute_strain(sc, use_roi=True, log=log)
    st = sc.state
    label = "with_roi" if use_roi else "without_roi"
    raw = np.asarray(st.strain_raw[label], float)
    assert raw.shape[1:] == (n, n), raw.shape
    valid = raw[4] > 0.5
    crystal = ~truth["vacuum"]
    n_fail = int((crystal & ~valid).sum())
    if (valid & ~crystal).any():
        raise RuntimeError("lattice fit reported valid pixels inside the vacuum band")
    if not valid.any():
        raise RuntimeError("no valid pixel: nothing to compare")

    ref_mask = E._reference_roi_mask(sc, st.strainmap_full.rshape) if use_roi else None
    d_yy = (raw[0] - S.expected_eyy(truth, ref_mask))[valid] * 100.0   # % strain
    exx = raw[1][valid] * 100.0                                          # truth: 0
    exy = raw[2][valid] * 100.0                                          # truth: 0
    q_fit = sc.params.q_px_fitted
    if q_fit is None:
        raise RuntimeError("q_px_fitted was not set by the calibration")
    card = (st.crystal_frame_cards or {}).get(label) or {}
    applied = float(card.get("x_angle_applied_deg", sc.params.coordinate_rotation))
    return {
        "bias_eyy_pct": float(np.mean(d_yy)),
        "rmse_eyy_pct": float(np.sqrt(np.mean(d_yy ** 2))),
        "maxabs_eyy_pct": float(np.max(np.abs(d_yy))),
        "p95_eyy_pct": float(np.percentile(np.abs(d_yy), 95)),
        "rms_exx_pct": float(np.sqrt(np.mean(exx ** 2))),
        "rms_exy_pct": float(np.sqrt(np.mean(exy ** 2))),
        "q_px_rel_err_pct": float((q_fit - S.Q_PIX) / S.Q_PIX * 100.0),
        "angle_err_deg": wrap_axial_deg(applied - S.PHI_DEG),
        "fit_fail_frac": n_fail / int(crystal.sum()),
        "n_valid": int(valid.sum()),
    }


# ── driver ────────────────────────────────────────────────────────────────────

def build_cases(cfg: dict) -> list[tuple[str, object, dict]]:
    return build_cases_generic(cfg, BASE_KEYS)


def build_cases_generic(cfg: dict, base_keys) -> list[tuple[str, object, dict]]:
    """Base case + one-factor sweeps ({name, set, vary, values}); every key is required."""
    BASE_KEYS = tuple(base_keys)  # noqa: N806  (shadow on purpose: same checks below)
    base = cfg["base"]
    missing = [k for k in BASE_KEYS if k not in base]
    extra = [k for k in base if k not in BASE_KEYS]
    if missing or extra:
        raise SystemExit(f"ABORT: config base missing {missing} / unknown {extra}")
    cases = [("base", None, dict(base))]
    for sw in cfg["sweeps"]:          # {"name", "set": {fixed overrides}, "vary", "values"}
        if "grid" in sw:              # full factorial: {"name", "set", "grid": {key: [values]}}
            import itertools
            gk = list(sw["grid"])
            bad = [k for k in [*gk, *sw["set"]] if k not in BASE_KEYS]
            if bad:
                raise SystemExit(f"ABORT: grid {sw['name']!r} has unknown keys {bad}")
            for combo in itertools.product(*(sw["grid"][k] for k in gk)):
                c = dict(base)
                c.update(sw["set"])
                c.update(dict(zip(gk, combo)))
                cases.append((sw["name"], dict(zip(gk, combo)), c))
            continue
        keys = [sw["vary"], *sw["set"]]
        bad = [k for k in keys if k not in BASE_KEYS]
        if bad:
            raise SystemExit(f"ABORT: sweep {sw['name']!r} has unknown keys {bad}")
        for v in sw["values"]:
            c = dict(base)
            c.update(sw["set"])
            c[sw["vary"]] = v
            cases.append((sw["name"], v, c))
    return cases


def main(cfg_path: str) -> None:
    cfg_file = Path(cfg_path).resolve()
    cfg = json.loads(cfg_file.read_text())
    p4 = check_py4dstem(cfg["reference"])
    env = environment(p4)
    seeds = list(range(int(cfg["seed0"]), int(cfg["seed0"]) + int(cfg["n_seeds"])))
    cases = build_cases(cfg)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / cfg["output_dir"] / f"{stamp}_{env['fast4d_commit'][:7]}_{platform.system()}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "env.json").write_text(json.dumps(env, indent=2))

    logging.basicConfig(filename=out / "run.log", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logging.captureWarnings(True)          # warnings kept (in run.log), not suppressed
    logger = logging.getLogger("bench")
    wcount = {"n": 0}
    _show = warnings.showwarning

    def _count(*a, **k):
        wcount["n"] += 1
        _show(*a, **k)
    warnings.showwarning = _count

    rows = []
    t0 = time.time()
    for factor, value, case in cases:
        for seed in seeds:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                m = run_case(case, seed, log=logger.info)
            if buf.getvalue().strip():
                logger.info("[stdout %s=%s seed=%d]\n%s", factor, value, seed, buf.getvalue())
            rows.append({"factor": factor, "value": json.dumps(value), "seed": seed, **m})

    with open(out / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for factor, value, _c in cases:
        rs = [r for r in rows if r["factor"] == factor and r["value"] == json.dumps(value)]
        s = {"factor": factor, "value": json.dumps(value), "n_seeds": len(rs)}
        for k in METRICS:
            v = np.array([r[k] for r in rs], float)
            s[k] = float(v.mean())
            s[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        summary.append(s)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (out / "summary.json").write_text(json.dumps(
        {"env": {k: env[k] for k in env if k != "packages"}, "warnings": wcount["n"],
         "elapsed_s": round(time.time() - t0, 1), "summary": summary}, indent=2))

    print(f"py4DSTEM {p4['version']} sha {p4['sha256'][:12]} | Fast4D {env['fast4d_commit'][:7]} "
          f"code {env['fast4d_code_sha256'][:12]} | {platform.system()} | seeds {len(seeds)} | "
          f"warnings {wcount['n']} | {time.time() - t0:.0f} s")
    print(f"{'factor':>14} {'value':>16} {'bias%':>8} {'RMSE%':>8} {'p95%':>7} {'max%':>7} "
          f"{'exx%':>7} {'exy%':>7} {'dq%':>8} {'dang':>7} {'fail':>5}")
    for s in summary:
        print(f"{s['factor']:>14} {s['value'][:16]:>16} {s['bias_eyy_pct']:8.4f} "
              f"{s['rmse_eyy_pct']:8.4f} {s['p95_eyy_pct']:7.3f} {s['maxabs_eyy_pct']:7.3f} "
              f"{s['rms_exx_pct']:7.3f} {s['rms_exy_pct']:7.3f} {s['q_px_rel_err_pct']:8.4f} "
              f"{s['angle_err_deg']:7.3f} {s['fit_fail_frac']:5.2f}")
    print(f"-> {out}")
    import plot_results                    # figures/errors_vs_factors.png
    plot_results.main(str(out))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/benchmark/benchmark_errors.py <config.json>")
    main(sys.argv[1])
