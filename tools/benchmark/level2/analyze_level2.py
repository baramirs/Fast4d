"""Summary, ranking and figures for a level-2 run folder (run_level2.py output).

Usage:
    python tools/benchmark/level2/analyze_level2.py benchmark_results/<run>_L2

Writes <run>/summary.csv (mean and SD over seeds per case), <run>/ranking.csv (cases by
mean total RMSE; any failed seed ranks the case last), <run>/main_effects.csv (mean total
RMSE per level of each grid factor) and figures in <run>/figures/.
Probe diameter (information only): d = 1.22 lambda / alpha, lambda(200 kV) = 2.508 pm.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
LAMBDA_PM = 2.508
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
METRICS = ["rmse_total", "exx_bias", "eyy_bias", "exy_bias", "exx_sigma", "eyy_sigma",
           "exy_sigma", "theta_rmse_deg", "theta_bias_deg", "frame_err_deg", "q_rel_err_pct",
           "pos_rms_px", "rate_111", "rate_220", "rate_004", "false_pos", "valid_frac"]


def preset_label(det: dict) -> str:
    sens = det["detect_min_absolute_intensity"] == 0 and det["detect_min_relative_intensity"] < 0.01
    return f"{det['detect_subpixel']} / {'sensitive' if sens else 'default'}"


def load(run: Path):
    cfg = json.loads((run / "config.json").read_text())
    rows = list(csv.DictReader(open(run / "per_seed.csv")))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["factor"], r["value"])].append(r)
    return cfg, groups


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def summarize(groups) -> list[dict]:
    out = []
    for (factor, value), rs in groups.items():
        ok = [r for r in rs if r["ok"] == "True"]
        s = {"factor": factor, "value": value, "n_seeds": len(rs), "n_ok": len(ok),
             "errors": "; ".join(sorted({r["error"] for r in rs if r["error"]}))[:200]}
        for m in METRICS:
            v = np.array([fnum(r.get(m)) for r in ok])
            v = v[np.isfinite(v)]
            s[m] = float(v.mean()) if len(v) else float("nan")
            s[m + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        q = {r.get("q_check", "") for r in rs} - {""}
        s["q_check"] = ",".join(sorted(q))
        out.append(s)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def stage_a(run: Path, cfg: dict, summ: list[dict]) -> None:
    grid_rows = [s for s in summ if s["factor"] != "base" and s["value"].startswith("{")]
    if not grid_rows:
        return
    for s in grid_rows:
        v = json.loads(s["value"])
        s["q_px"], s["alpha_mrad"], s["dose"] = v["q_px"], v["alpha_mrad"], v["dose"]
        s["preset"] = preset_label(v["detect"])
        s["probe_nm"] = 1.22 * LAMBDA_PM / v["alpha_mrad"]       # pm / mrad = nm
        s["failed_seeds"] = s["n_seeds"] - s["n_ok"]
    rank = sorted(grid_rows, key=lambda s: (s["failed_seeds"] > 0, s["rmse_total"]
                                            if np.isfinite(s["rmse_total"]) else 1e9))
    cols = ["q_px", "alpha_mrad", "probe_nm", "dose", "preset", "n_ok", "n_seeds", "rmse_total",
            "rmse_total_sd", "exx_bias", "eyy_bias", "exy_bias", "eyy_sigma", "theta_rmse_deg",
            "q_rel_err_pct", "pos_rms_px", "rate_111", "rate_220", "rate_004", "errors"]
    write_csv(run / "ranking.csv", [{k: s.get(k) for k in cols} for s in rank])
    eff = []
    for f in ("q_px", "alpha_mrad", "dose", "preset"):
        for lvl in sorted({s[f] for s in grid_rows}, key=str):
            sel = [s for s in grid_rows if s[f] == lvl]
            ok = [s["rmse_total"] for s in sel if s["failed_seeds"] == 0]
            eff.append({"factor": f, "level": lvl, "n_cases": len(sel), "n_failed_cases":
                        sum(s["failed_seeds"] > 0 for s in sel),
                        "mean_rmse_ok_cases": float(np.mean(ok)) if ok else float("nan"),
                        "median_rmse_ok_cases": float(np.median(ok)) if ok else float("nan")})
    write_csv(run / "main_effects.csv", eff)

    qs = sorted({s["q_px"] for s in grid_rows}, reverse=True)
    presets = sorted({s["preset"] for s in grid_rows})
    alphas = sorted({s["alpha_mrad"] for s in grid_rows})
    doses = sorted({s["dose"] for s in grid_rows})
    vals = np.array([s["rmse_total"] for s in grid_rows if s["failed_seeds"] == 0])
    vmin, vmax = (float(np.nanmin(vals)), float(np.nanmax(vals))) if len(vals) else (0.01, 1)
    from matplotlib.colors import ListedColormap, LogNorm
    cmap = ListedColormap(SEQ)
    norm = LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, vmin * 1.01))
    fig, axs = plt.subplots(len(presets), len(qs), figsize=(4.2 * len(qs), 2.7 * len(presets)),
                            facecolor=SURF, squeeze=False)
    for i, pr in enumerate(presets):
        for j, q in enumerate(qs):
            ax = axs[i, j]
            ax.set_facecolor(SURF)
            m = np.full((len(alphas), len(doses)), np.nan)
            fail = np.zeros_like(m, bool)
            for s in grid_rows:
                if s["preset"] == pr and s["q_px"] == q:
                    a, d = alphas.index(s["alpha_mrad"]), doses.index(s["dose"])
                    m[a, d] = s["rmse_total"]
                    fail[a, d] = s["failed_seeds"] > 0
            ax.imshow(np.where(fail, np.nan, m), cmap=cmap, norm=norm, origin="lower", aspect="auto")
            for a in range(len(alphas)):
                for d in range(len(doses)):
                    txt = "fail" if fail[a, d] else (f"{m[a, d]:.3f}" if np.isfinite(m[a, d]) else "")
                    dark = np.isfinite(m[a, d]) and not fail[a, d] and norm(m[a, d]) > 0.55
                    ax.text(d, a, txt, ha="center", va="center", fontsize=8,
                            color="#fcfcfb" if dark else INK)
            ax.set_xticks(range(len(doses)), [f"{d:.0e}" for d in doses], fontsize=7, color=INK2)
            ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas], fontsize=7, color=INK2)
            ax.set_xlabel("dose (e / pattern)", fontsize=8, color=INK2)
            ax.set_ylabel("alpha (mrad)", fontsize=8, color=INK2)
            ax.set_title(f"q_px {q:g} 1/A/px | {pr}", fontsize=9, color=INK, loc="left")
            for sp in ax.spines.values():
                sp.set_color(AXIS)
    fig.suptitle("Level 2: total strain RMSE vs truth (% strain, mean of seeds; lower is better)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    (run / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(run / "figures" / f"stageA_rmse_heatmaps.{ext}", dpi=170, facecolor=SURF)
    plt.close(fig)


def stage_b(run: Path, summ: list[dict]) -> None:
    sweeps = [f for f in dict.fromkeys(s["factor"] for s in summ) if f != "base"
              and not any(x["value"].startswith("{") for x in summ if x["factor"] == f)]
    if not sweeps:
        return
    base = next(s for s in summ if s["factor"] == "base")
    panels = [("rmse_total", "total RMSE (%)"), ("exy_bias", "exy bias (%)"),
              ("theta_rmse_deg", "theta RMSE (deg)"), ("frame_err_deg", "frame error (deg)")]
    fig, axs = plt.subplots(len(sweeps), len(panels), figsize=(13, 2.3 * len(sweeps)),
                            facecolor=SURF, squeeze=False)
    for i, f in enumerate(sweeps):
        rs = [s for s in summ if s["factor"] == f]
        labels = [s["value"] for s in rs]
        for j, (m, lab) in enumerate(panels):
            ax = axs[i, j]
            ax.set_facecolor(SURF)
            y = np.array([s[m] for s in rs], float)
            e = np.nan_to_num(np.array([s[m + "_sd"] for s in rs], float))
            x = np.arange(len(rs))
            ax.bar(x, y, yerr=e, color="#2a78d6", width=0.6, capsize=2, ecolor=INK2)
            ax.axhline(base[m], color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.axhline(0, color=AXIS, lw=0.8)
            ax.set_xticks(x, labels, fontsize=7, color=INK2, rotation=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK2)
            if i == 0:
                ax.set_title(lab, fontsize=9, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(f, fontsize=8, color=INK2)
    fig.suptitle("Level 2 rotation tests (bars: mean +- SD of seeds; dashed: base case)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    (run / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(run / "figures" / f"stageB_rotation.{ext}", dpi=170, facecolor=SURF)
    plt.close(fig)


def main(run_dir: str) -> None:
    run = Path(run_dir).resolve()
    cfg, groups = load(run)
    summ = summarize(groups)
    write_csv(run / "summary.csv", summ)
    stage_a(run, cfg, summ)
    stage_b(run, summ)
    print(f"-> {run}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/benchmark/level2/analyze_level2.py <run_dir>")
    main(sys.argv[1])
