"""Fig. 2: training process and homology-controlled benchmark.

a     training process -- validation F1 / MCC and the train/validation loss per
      epoch, with the selected checkpoint (epoch 3) marked
b     F1 across the four similarity cutoffs, all six methods
c-e   Balanced ACC / MCC / Recall with the cutoffs merged
      (bar = mean over the four cutoffs, whisker = min..max)

Source data: data/fig2_benchmark.tsv holds the metric values of the six
methods at the four cutoffs as reported in the manuscript (the CysForcing
rows are recomputed by reproduce/eval_homology.py), data/fig2_training.tsv
the per-epoch training log of the released model.

    python paper/fig2.py   # -> paper/out/fig2.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
# ESM-2 family in greys, SaProt family in blues, ours in red
METHODS = [("ESM2-650M", "#C2C9D0"), ("ESM2-3B", "#949EA8"), ("ESM2-15B", "#6E7781"),
           ("SaProt-650M", "#A9CBE0"), ("SaProt-1B", "#6FA8C9"), ("CysForcing", CYS)]
SHORT = ["ESM2\n650M", "ESM2\n3B", "ESM2\n15B", "SaProt\n650M", "SaProt\n1B", "Cys\nForcing"]
CUTS = [100, 80, 60, 40]


def finish(ax, ylabel=None, ylim=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E6E9EC", lw=0.45)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=5.8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.4, fontweight="bold", labelpad=3)


def tag(fig, ax, letter, dx=0.050, dy=0.17):
    pos = ax.get_position()
    fig.text(pos.x0 - dx, pos.y1 + dy * pos.height, letter, fontsize=9.5,
             fontweight="bold", va="top", ha="left", color="#111111")


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "fig2_benchmark.tsv"), sep="\t")
    tr = pd.read_csv(os.path.join(HERE, "data", "fig2_training.tsv"), sep="\t")

    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    gsA = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.75], wspace=0.30,
                           left=0.072, right=0.985, top=0.935, bottom=0.595)
    gsB = fig.add_gridspec(1, 3, wspace=0.30,
                           left=0.072, right=0.985, top=0.455, bottom=0.145)
    axes = []

    # ---- a: training process ----
    ax = fig.add_subplot(gsA[0, 0])
    e = tr["epoch"].to_numpy()
    ax.plot(e, tr["val_f1"], "-o", color=CYS, lw=1.5, ms=2.8, mew=0,
            zorder=6, label="Val F1")
    ax.plot(e, tr["val_mcc"], "-s", color="#6FA8C9", lw=1.2, ms=2.4, mew=0,
            zorder=5, label="Val MCC")
    finish(ax, "Validation score", (0, 0.62))
    ax.set_xlabel("Epoch", fontsize=6.8)
    ax.set_xticks(e[::2])

    ax2 = ax.twinx()
    ax2.plot(e, tr["train_loss"], "--", color="#8E99A4", lw=1.1, zorder=3,
             label="Train loss")
    ax2.set_ylim(0, 3.0)
    ax2.set_ylabel("Train loss", fontsize=7.0, color="#5B6670", labelpad=2)
    ax2.tick_params(labelsize=5.6, colors="#5B6670")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#B6BDC4")

    best = int(tr.loc[tr["val_f1"].idxmax(), "epoch"])
    ax.axvline(best, color="#2B3137", lw=0.6, ls=(0, (2.5, 2)), zorder=2)
    ax.annotate(f"selected\nepoch {best}", xy=(best, 0.60), xytext=(best + 0.45, 0.60),
                fontsize=5.4, color="#2B3137", va="top", ha="left", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.5, loc="lower right", ncol=1,
              handlelength=1.3, columnspacing=0.9, borderpad=0.2)
    axes.append(ax)

    # ---- b: F1 across all four cutoffs ----
    ax = fig.add_subplot(gsA[0, 1])
    x = np.arange(len(CUTS))
    w = 0.82 / len(METHODS)
    for q, (m, col) in enumerate(METHODS):
        vals = [float(df[(df.cutoff == c) & (df.method == m)]["f1"].iloc[0]) for c in CUTS]
        lead = m == "CysForcing"
        ax.bar(x - 0.41 + w * (q + 0.5), vals, w * 0.9, color=col, zorder=3,
               hatch="///" if lead else None,
               edgecolor="white" if lead else "none", linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CUTS], fontsize=5.8)
    ax.set_xlabel("Max. sequence similarity (%)", fontsize=6.8)
    finish(ax, "F1", (0, 1))
    axes.append(ax)

    # ---- c-e: cutoffs merged ----
    for k, (metric, lbl, ymax) in enumerate([("bacc", "Balanced ACC", 1.0),
                                             ("mcc", "MCC", 1.0),
                                             ("recall", "Recall", 1.0)]):
        ax = fig.add_subplot(gsB[0, k])
        mean, lo, hi = [], [], []
        for m, _ in METHODS:
            v = np.array([float(df[(df.cutoff == c) & (df.method == m)][metric].iloc[0])
                          for c in CUTS])
            mean.append(v.mean()); lo.append(v.mean() - v.min()); hi.append(v.max() - v.mean())
        xx = np.arange(len(METHODS))
        for q, (m, col) in enumerate(METHODS):
            lead = m == "CysForcing"
            ax.bar(xx[q], mean[q], 0.70, color=col, zorder=3,
                   hatch="///" if lead else None,
                   edgecolor="white" if lead else "none", linewidth=0)
        ax.errorbar(xx, mean, yerr=[lo, hi], fmt="none", ecolor="#2B3137",
                    elinewidth=0.55, capsize=1.4, capthick=0.55, zorder=4)
        for q, v in enumerate(mean):
            ax.text(xx[q], v + hi[q] + ymax * 0.03, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=4.4, color="#2B3137", zorder=5,
                    fontweight="bold" if METHODS[q][0] == "CysForcing" else "normal")
        ax.set_xticks(xx)
        ax.set_xticklabels(SHORT, fontsize=4.8)
        finish(ax, lbl, (0, ymax))
        axes.append(ax)

    fig.canvas.draw()
    for ax, letter in zip(axes, "abcde"):
        tag(fig, ax, letter)

    handles = []
    for m, col in METHODS:
        lead = m == "CysForcing"
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=col,
                                     hatch="///" if lead else None,
                                     edgecolor="white" if lead else "none", linewidth=0))
    fig.legend(handles, [m for m, _ in METHODS], loc="lower center", ncol=6,
               fontsize=6.3, bbox_to_anchor=(0.53, 0.004), columnspacing=1.6,
               handlelength=1.4, handletextpad=0.5)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(HERE, "out", f"fig2.{ext}")
        fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
