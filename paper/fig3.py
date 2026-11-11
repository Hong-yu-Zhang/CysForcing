"""Fig. 3: ablation and representation analysis.

a      MCC across the four similarity cutoffs (kept in full -- carries the
       "gains grow as homology drops" message)
b-d    Balanced ACC / F1 / Recall with the similarity cutoffs merged
       (bar = mean over the four cutoffs, whisker = min..max)
e      bootstrap distribution of the ratio-ordering score rho, against a
       permuted-ratio null
f      PCA of the Cys-site embeddings coloured by local mean reactivity ratio

Source data: data/fig3_ablation.tsv (metric values as reported in the
manuscript) and data/fig3_interpret.npz (fused per-Cys embeddings of the
held-out sites, the rho bootstrap and its permuted-ratio null).

    python paper/fig3.py   # -> paper/out/fig3.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.2,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#2B3137",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 1.9,
    "ytick.major.size": 1.9,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "hatch.linewidth": 0.45,
})

CYS = "#C0463C"
NULL = "#C3C9CF"
# grey -> blue -> green ramp, with the full model in the Fig. 4 red
METHODS = [("ESM2-650m", "#B8BFC6"), ("Saprot-650m", "#8E99A4"),
           ("Baseline", "#6FA8C9"), ("+ CL", "#6E9E63"), ("+ CysForcing", CYS)]
SHORT = ["ESM2", "SaProt", "Base", "+CL", "+Cys"]
CUTS = [100, 80, 60, 40]


def finish(ax, ylabel=None, ylim=(0, 1)):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E6E9EC", lw=0.45)
    ax.set_axisbelow(True)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=5.8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.4, fontweight="bold", labelpad=3)


def tag(fig, ax, letter, dx=0.052, dy=0.16):
    pos = ax.get_position()
    fig.text(pos.x0 - dx, pos.y1 + dy * pos.height, letter, fontsize=9.5,
             fontweight="bold", va="top", ha="left", color="#111111")


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "fig3_ablation.tsv"), sep="\t")
    iz = np.load(os.path.join(HERE, "data", "fig3_interpret.npz"))

    fig = plt.figure(figsize=(183 * MM, 125 * MM))
    gsA = fig.add_gridspec(1, 4, width_ratios=[2.25, 1, 1, 1], wspace=0.40,
                           left=0.075, right=0.985, top=0.935, bottom=0.605)
    gsB = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.48, 1.48], wspace=0.22,
                           left=0.075, right=0.925, top=0.455, bottom=0.135)
    axes = []

    # ---- a: full cutoff breakdown ----
    ax = fig.add_subplot(gsA[0, 0])
    x = np.arange(len(CUTS))
    w = 0.8 / len(METHODS)
    for q, (m, col) in enumerate(METHODS):
        vals = [float(df[(df.cutoff == c) & (df.method == m)]["f1"].iloc[0]) for c in CUTS]
        lead = m == "+ CysForcing"
        ax.bar(x - 0.4 + w * (q + 0.5), vals, w * 0.9, color=col, zorder=3,
               hatch="///" if lead else None,
               edgecolor="white" if lead else "none", linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CUTS], fontsize=5.8)
    ax.set_xlabel("Max. sequence similarity (%)", fontsize=6.6)
    # F1 leads panel a: on the published numbers CysForcing is ahead at every
    # cutoff (+0.041..+0.068), and it has the widest dynamic range.
    finish(ax, "F1", ylim=(0, 1))
    axes.append(ax)

    # ---- b-d: cutoffs merged ----
    for k, (metric, lbl, ymax) in enumerate([("mcc", "MCC", 1.0),
                                             ("bacc", "Balanced ACC", 1.0),
                                             ("recall", "Recall", 1.0)]):
        ax = fig.add_subplot(gsA[0, k + 1])
        mean, lo, hi = [], [], []
        for m, _ in METHODS:
            v = np.array([float(df[(df.cutoff == c) & (df.method == m)][metric].iloc[0])
                          for c in CUTS])
            mean.append(v.mean()); lo.append(v.mean() - v.min()); hi.append(v.max() - v.mean())
        xx = np.arange(len(METHODS))
        for q, (m, col) in enumerate(METHODS):
            lead = m == "+ CysForcing"
            ax.bar(xx[q], mean[q], 0.68, color=col, zorder=3,
                   hatch="///" if lead else None,
                   edgecolor="white" if lead else "none", linewidth=0)
        ax.errorbar(xx, mean, yerr=[lo, hi], fmt="none", ecolor="#2B3137",
                    elinewidth=0.55, capsize=1.4, capthick=0.55, zorder=4)
        for q, v in enumerate(mean):
            ax.text(xx[q], v + hi[q] + ymax * 0.03, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=4.4,
                    color="#2B3137", zorder=5,
                    fontweight="bold" if METHODS[q][0] == "+ CysForcing" else "normal")
        ax.set_xticks(xx)
        ax.set_xticklabels(SHORT, fontsize=5.4, rotation=35, ha="right")
        finish(ax, lbl, ylim=(0, ymax))
        axes.append(ax)

    # ---- e: ratio-ordering score ----
    ax = fig.add_subplot(gsB[0, 0])
    grid = np.linspace(-0.12, 0.62, 400)
    nullv = np.concatenate([iz["raw_perm"], iz["both_align_perm"]])
    entries = [("Permuted-ratio null", nullv, NULL, None),
               ("Baseline", iz["raw_boot"], "#6FA8C9", float(iz["raw_rho"])),
               ("+ CysForcing", iz["both_align_boot"], CYS, float(iz["both_align_rho"]))]
    top = 0.0
    for name, data, col, r in entries:
        dens = gaussian_kde(data)(grid)
        ax.fill_between(grid, dens, color=col, alpha=0.45, lw=0, zorder=3)
        ax.plot(grid, dens, color=col, lw=1.1, zorder=4)
        top = max(top, float(dens.max()))
        if r is not None:
            ax.text(r, float(dens.max()) * 1.04, f"ρ = {r:.2f}", ha="center",
                    va="bottom", fontsize=5.9, color=col, fontweight="bold", zorder=6)
    ax.set_ylim(0, top * 1.24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.12, 0.62)
    ax.set_xlabel("Ratio-ordering score ρ", fontsize=6.8)   # "2,000 bootstraps" -> caption
    ax.set_ylabel("Density", fontsize=7.4, fontweight="bold", labelpad=3)
    ax.set_yticks([])
    ax.tick_params(labelsize=5.8)
    ax.legend(handles=[Line2D([0], [0], color=NULL, lw=3, label="Permuted-ratio null"),
                       Line2D([0], [0], color="#6FA8C9", lw=3, label="Baseline"),
                       Line2D([0], [0], color=CYS, lw=3, label="+ CysForcing")],
              fontsize=5.6, loc="upper left", handlelength=1.1, borderpad=0.2)
    axes.append(ax)

    # ---- f: PCA of the embeddings ----
    allr = np.concatenate([iz["raw_local"], iz["both_align_local"]])
    norm = LogNorm(vmin=max(np.percentile(allr, 2), 1e-3), vmax=np.percentile(allr, 98))
    sc = None
    for k, (key, name) in enumerate([("raw", "Baseline"), ("both_align", "+ CysForcing")]):
        ax = fig.add_subplot(gsB[0, k + 1])
        xy = iz[f"{key}_xy"]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=iz[f"{key}_local"], cmap="RdYlBu",
                        norm=norm, s=4.2, linewidths=0, rasterized=True)
        # clip the few extreme points so the cloud fills the frame
        (x0, x1), (y0, y1) = (np.percentile(xy[:, c], [0.6, 99.4]) for c in (0, 1))
        ax.set_xlim(x0 - (x1 - x0) * 0.025, x1 + (x1 - x0) * 0.025)
        ax.set_ylim(y0 - (y1 - y0) * 0.025, y1 + (y1 - y0) * 0.025)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#AEB6BE"); sp.set_linewidth(0.8)
        ax.set_title(f"{name}   (ρ = {float(iz[key + '_rho']):.2f})",
                     fontsize=7.0, fontweight="bold", pad=3.5,
                     color=CYS if key == "both_align" else "#2B3137")
        axes.append(ax)
    cax = fig.add_axes([0.937, 0.155, 0.013, 0.26])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Local mean ratio", fontsize=6.4, labelpad=3)
    cb.ax.tick_params(labelsize=5.4, width=0.5, length=1.8)
    cb.outline.set_linewidth(0.5)

    fig.canvas.draw()
    for ax, letter in zip(axes, "abcdeff"):
        if letter == "f" and ax is axes[-1]:
            continue
        tag(fig, ax, letter)

    handles = []
    for m, col in METHODS:
        lead = m == "+ CysForcing"
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=col,
                                     hatch="///" if lead else None,
                                     edgecolor="white" if lead else "none", linewidth=0))
    fig.legend(handles, [m for m, _ in METHODS], loc="lower center", ncol=5,
               fontsize=6.4, bbox_to_anchor=(0.5, 0.005), columnspacing=1.8,
               handlelength=1.4, handletextpad=0.5)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(HERE, "out", f"fig3.{ext}")
        fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
