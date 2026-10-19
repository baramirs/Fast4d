"""Figures for a benchmark results folder (protocol: docs/PROTOCOLO_BENCHMARK_ERRORES.md).

Usage:
    python tools/benchmark/plot_results.py benchmark_results/<run_dir>

Writes <run_dir>/figures/errors_vs_factors.png (+ .pdf): one panel per sweep with
RMSE of e_yy and spurious RMS e_xy versus the swept value (mean +- SD over seeds), and
the base case as a dashed reference line.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False   # plain hyphen-minus in labels
import numpy as np

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_EYY, C_EXY = "#2a78d6", "#eb6834"           # categorical slots 1, 2 (validated)
LABELS = {
    "jitter_px": "Peak position noise (px)",
    "e_max": "Strain amplitude e_max (fraction)",
    "rot_err_manual": "Frame rotation error, manual mode (deg)",
    "center_err_px": "Initial centre error (px)",
    "q_guess": "Initial Q pixel guess (1/A per px; true 0.020)",
    "ellipse_uncorr": "Uncorrected ellipse a/b (theta 25 deg)",
    "roi_vacuum": "Reference ROI, 6-column vacuum band",
}


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main(run_dir: str) -> None:
    run = Path(run_dir).resolve()
    rows = list(csv.DictReader(open(run / "summary.csv")))
    env = json.loads((run / "env.json").read_text())
    base = next(r for r in rows if r["factor"] == "base")
    seeds = list(csv.DictReader(open(run / "per_seed.csv")))
    sweeps = [f for f in dict.fromkeys(r["factor"] for r in rows) if f != "base"]

    ncol = 2
    nrow = int(np.ceil(len(sweeps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.9 * nrow), facecolor="#fcfcfb")
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, sweeps):
        style(ax)
        rs = [r for r in rows if r["factor"] == f]
        vals = [json.loads(r["value"]) for r in rs]
        numeric = all(isinstance(v, (int, float)) for v in vals)
        x = np.array(vals, float) if numeric else np.arange(len(vals))
        for key, col, mk, lab in (("rmse_eyy_pct", C_EYY, "o", "RMSE e_yy"),
                                  ("rms_exy_pct", C_EXY, "s", "RMS e_xy (spurious)")):
            y = np.array([float(r[key]) for r in rs])
            sd = np.nan_to_num(np.array([float(r[key + "_sd"]) for r in rs]))
            ax.errorbar(x, y, yerr=sd, color=col, marker=mk, ms=5, lw=2, capsize=3,
                        mec="#fcfcfb", mew=1.5, label=lab, zorder=3)
            ax.axhline(float(base[key]), color=col, lw=1, ls=(0, (4, 3)), alpha=0.6, zorder=2)
            for xi, r in zip(x, rs):          # every seed, so bimodal failures stay visible
                ys = [float(q[key]) for q in seeds if q["factor"] == f and q["value"] == r["value"]]
                ax.scatter(np.full(len(ys), xi), ys, s=14, facecolor="none", edgecolor=col,
                           lw=0.9, alpha=0.8, zorder=4)
        if not numeric:
            names = {"ellipse_uncorr": lambda v: f"a/b = {v[0]:g}",
                     "roi_vacuum": lambda v: f"ROI {v}"}
            ax.set_xticks(x, [names.get(f, str)(v) for v in vals])
        ax.set_xlabel(LABELS.get(f, f), color=INK2, fontsize=9)
        ax.set_ylabel("% strain", color=INK2, fontsize=9)
        ax.set_ylim(bottom=0)
    for ax in axes[len(sweeps):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=9,
               labelcolor=INK2, ncol=2)
    p4 = env["py4dstem"]
    fig.suptitle(f"Fast4D level-1 error vs ground truth (line: mean ± SD over seeds; open marks: each seed; dashed: base case)\n"
                 f"py4DSTEM {p4['version']} ({p4['sha256'][:12]}), Fast4D {env['fast4d_commit'][:7]}, "
                 f"{env['platform'].split('-')[0]}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    out = run / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"errors_vs_factors.{ext}", dpi=200, facecolor=fig.get_facecolor())
    print(f"-> {out / 'errors_vs_factors.png'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/benchmark/plot_results.py <run_dir>")
    main(sys.argv[1])
