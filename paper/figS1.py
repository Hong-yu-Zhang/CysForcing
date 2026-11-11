"""Supplementary Fig. S1: sensitivity of CysForcing to the sequence backbone.

The default frozen ESM-2 650M encoder is replaced by ESM-C 300M / 600M and the
model is retrained under the identical recipe (seed 42; reproduce/run_backbone.sh).

  a  AUROC at the 100/80/60/40 % similarity cutoffs (homology-controlled test set)
  b  MCC
  c  F1

Source data: data/figS1_backbones.tsv, the output of reproduce/eval_homology.py
for each backbone. The external CIAA AF2-denovo numbers are in Supplementary
Table S1.

    python paper/figS1.py   # -> paper/out/figS1.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

CUTOFFS = [100, 80, 60, 40]
BACKBONES = ["ESM-2 650M", "ESM-C 300M", "ESM-C 600M"]
STYLE = {
    "ESM-2 650M": dict(color="#c9302c", edgecolor="#7a1d1a", hatch="///"),
    "ESM-C 300M": dict(color="#9ecae1", edgecolor="#2a4d69", hatch=None),
    "ESM-C 600M": dict(color="#4292c6", edgecolor="#1f4e79", hatch=None),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.9,
    "mathtext.default": "regular",
})


def load():
    df = pd.read_csv(os.path.join(HERE, "data", "figS1_backbones.tsv"), sep="\t")
    out = {}
    for metric, col in (("AUROC", "auroc"), ("MCC", "mcc"), ("F1", "f1")):
        # bars are drawn at the three-decimal values printed on them
        out[metric] = {b: [round(float(df[(df.backbone == b) & (df.cutoff == c)][col].iloc[0]), 3)
                           for c in CUTOFFS] for b in BACKBONES}
    return out


def draw(ax, data, ylabel, *, ylim, value_fontsize=7.4):
    x = np.arange(len(CUTOFFS))
    width = 0.24
    for k, b in enumerate(BACKBONES):
        off = (k - 1) * width
        vals = data[b]
        st = STYLE[b]
        ax.bar(x + off, vals, width, label=b, linewidth=0.6, zorder=3,
               color=st["color"], edgecolor=st["edgecolor"], hatch=st["hatch"])
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.006 * (ylim[1] - ylim[0]) / 0.45,
                    f"{v:.3f}", ha="center", va="bottom", rotation=90,
                    fontsize=value_fontsize,
                    fontweight=("bold" if b == "ESM-2 650M" else "normal"),
                    color="black", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}%" for c in CUTOFFS], fontsize=10.5)
    ax.set_xlabel("Seq. similarity (%)", fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel + r" $\uparrow$", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9.5)


def main():
    data = load()
    fig = plt.figure(figsize=(13.5, 4.4))
    gs = gridspec.GridSpec(1, 3, wspace=0.2, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw(axes[0], data["AUROC"], "AUROC", ylim=(0.0, 0.90))
    draw(axes[1], data["MCC"], "MCC", ylim=(0.0, 0.44))
    draw(axes[2], data["F1"], "F1", ylim=(0.0, 0.66))
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
                fontsize=16, fontweight="bold", ha="left", va="bottom")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[b]["color"],
                             edgecolor=STYLE[b]["edgecolor"], hatch=STYLE[b]["hatch"],
                             label=(b + " (default)" if b == "ESM-2 650M" else b))
               for b in BACKBONES]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=11)
    fig.subplots_adjust(bottom=0.2)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(HERE, "out", f"figS1.{ext}")
        fig.savefig(p, dpi=600, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
