"""Fig. 4: cross-method, cross-species zero-shot generalization.

Row 1  a-c  radar summary (AUROC, AUPRC, MCC, balanced accuracy) per OOD set
Row 2  d-f  AUROC split by homology-transfer regime
Row 3  g-i  MCC   split by homology-transfer regime

Every radar axis carries the metric name, the CysForcing value and its margin
over the best released predictor; all axes share one 0-1 scale.

    python paper/fig4_metrics.py   # writes the tables this script reads
    python paper/fig4.py           # -> paper/out/fig4.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
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
CIAARF = "#9AA3AB"
HYP = "#6FA8C9"
HOMO = "#E0A24B"
SBPCR = "#6E9E63"
REF = "#5B6670"
GREY_TXT = "#8C949B"
DOWN = "#4A7FA5"

DATASETS = [("Mouse", "Mouse (zero-shot)"),
            ("SL1344", "SL1344 (zero-shot)"),
            ("CIAA", "CIAA AF2-denovo")]
# (name, colour, line width, marker size); CysForcing is drawn last, on top
RADAR_METHODS = [("CIAA RF", CIAARF, 1.0, 2.4),
                 ("HyperCys", HYP, 1.0, 2.4),
                 ("sbPCR", SBPCR, 1.0, 2.4),
                 ("CysForcing", CYS, 1.7, 3.4)]
AXES = [("auroc", "AUROC"), ("auprc", "AUPRC"), ("mcc", "MCC"), ("bacc", "Balanced ACC")]
REGIMES = [("transferable", "Transferable"), ("non-transferable", "Non-transferable")]
METHODS5 = [("CysForcing", CYS), ("CIAA RF", CIAARF), ("HyperCys", HYP),
            ("sbPCR", SBPCR), ("Homology transfer", HOMO)]
RMAX = 1.0


def radar(ax, sub, title):
    n = len(AXES)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed = ang + ang[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    others = sub[sub.method != "CysForcing"]
    ref = {k: float(sub[sub.method == "CysForcing"][k].iloc[0]) for k, _ in AXES}
    best_other = {k: float(others[k].max()) for k, _ in AXES}

    for name, col, lw, ms in RADAR_METHODS:
        row = sub[sub.method == name]
        vals = [float(row[k].iloc[0]) for k, _ in AXES]
        v = vals + vals[:1]
        lead = name == "CysForcing"
        if lead:
            ax.fill(closed, v, color=col, alpha=0.16, zorder=4, linewidth=0)
        ax.plot(closed, v, color=col, ls="-", lw=lw, zorder=8 if lead else 5,
                solid_joinstyle="round", clip_on=False)
        ax.scatter(ang, vals, s=ms ** 2, marker="o", zorder=9 if lead else 6,
                   facecolors=col, edgecolors=col, linewidths=0, clip_on=False)

    # metric name, CysForcing value and margin over the best released predictor
    for a, (k, label) in zip(ang, AXES):
        phi = np.pi / 2 - a
        ox, oy = np.cos(phi) * 13.0, np.sin(phi) * 13.0
        d = ref[k] - best_other[k]
        dcol, dtxt = (CYS, f"+{d:.3f}") if d >= 0 else (DOWN, f"{d:.3f}")
        for dy, txt, fs, col, weight, style in [
                (7.0, label, 6.2, "#1F2328", "bold", "normal"),
                (0.8, f"{ref[k]:.3f}", 5.4, GREY_TXT, "normal", "italic"),
                (-5.6, dtxt, 5.2, dcol, "bold", "normal")]:
            ax.annotate(txt, xy=(a, RMAX), xycoords="data",
                        textcoords="offset points", xytext=(ox, oy + dy),
                        ha="center", va="center", fontsize=fs, color=col,
                        fontweight=weight, fontstyle=style, annotation_clip=False)

    ax.set_xticks(ang)
    ax.set_xticklabels([])
    ax.set_ylim(0, RMAX)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=4.6, color="#B6BDC3")
    ax.set_rlabel_position(45)
    ax.grid(color="#DEE3E7", lw=0.45)
    ax.spines["polar"].set_color("#E4E9ED")
    ax.spines["polar"].set_linewidth(0.5)
    ax.set_title(f"{title}   (n = {int(sub['n'].iloc[0]):,})", fontsize=6.9,
                 fontweight="bold", pad=30, color="#111111")


def regime_bars(ax, sub, mkey, ref):
    lo, hi = 0.0, 1.0          # d-i share one 0-1 axis for comparability
    ns = {rg: int(sub[sub.regime == rg]["n"].iloc[0]) for rg, _ in REGIMES}
    x = np.arange(len(REGIMES))
    w = 0.78 / len(METHODS5)
    for q, (mname, col) in enumerate(METHODS5):
        vals = [float(sub[(sub.regime == rg) & (sub.method == mname)][mkey].iloc[0])
                for rg, _ in REGIMES]
        pos = x - 0.39 + w * (q + 0.5)
        lead = mname == "CysForcing"
        ax.bar(pos, vals, w * 0.88, color=col, zorder=3, hatch="///" if lead else None,
               edgecolor="white" if lead else "none", linewidth=0)
        for xi, v in zip(pos, vals):
            ax.text(xi, max(v, 0.0) + (hi - lo) * 0.022, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=4.0, color="#2B3137", zorder=5,
                    fontweight="bold" if lead else "normal")
    ax.axhline(ref, color=REF, lw=0.6, ls=(0, (2.5, 2)), zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb}\n(n = {ns[rg]})" for rg, lb in REGIMES], fontsize=5.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E6E9EC", lw=0.45)
    ax.set_axisbelow(True)
    ax.set_ylim(lo, hi)
    ax.tick_params(labelsize=5.8)


def main():
    rad = pd.read_csv(os.path.join(OUT, "fig4_radar.tsv"), sep="\t")
    hom = pd.read_csv(os.path.join(OUT, "fig4_homology.tsv"), sep="\t")

    fig = plt.figure(figsize=(183 * MM, 164 * MM))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.30, 1.0, 1.0], hspace=0.26, wspace=0.26,
                           left=0.085, right=0.985, top=0.965, bottom=0.105)
    grid = [[None] * 3 for _ in range(3)]

    for j, (key, title) in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, j], polar=True)
        pos = ax.get_position()
        s = 0.66
        cx, cy = pos.x0 + pos.width / 2, pos.y0 + pos.height / 2
        ax.set_position([cx - pos.width * s / 2, cy - pos.height * s / 2,
                         pos.width * s, pos.height * s])
        radar(ax, rad[rad.dataset == key], title)
        grid[0][j] = ax

    for r, (mkey, mlabel, ref) in enumerate([("auroc", "AUROC", 0.5), ("mcc", "MCC", 0.0)]):
        for j, (key, _) in enumerate(DATASETS):
            ax = fig.add_subplot(gs[r + 1, j])
            regime_bars(ax, hom[hom.dataset == key], mkey, ref)
            if j == 0:
                ax.set_ylabel(mlabel, fontsize=7.6, fontweight="bold", labelpad=3)
            grid[r + 1][j] = ax

    fig.canvas.draw()
    for j in range(3):
        lx = grid[1][j].get_position().x0 - 0.052
        for r in range(3):
            pos = grid[r][j].get_position()
            y = pos.y1 + (0.42 if r == 0 else 0.16) * pos.height
            fig.text(lx, y, "abcdefghi"[r * 3 + j], fontsize=9.5, fontweight="bold",
                     va="top", ha="left", color="#111111")

    handles = [Line2D([0], [0], color=CYS, lw=1.7, marker="o", ms=3.4, mew=0),
               Line2D([0], [0], color=CIAARF, lw=1.0, marker="o", ms=2.4, mew=0),
               Line2D([0], [0], color=HYP, lw=1.0, marker="o", ms=2.4, mew=0),
               Line2D([0], [0], color=SBPCR, lw=1.0, marker="o", ms=2.4, mew=0),
               Line2D([0], [0], color=HOMO, lw=3.4)]
    labels = ["CysForcing (ours)", "CIAA RF", "HyperCys", "sbPCR", "Homology transfer"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.5,
               bbox_to_anchor=(0.535, 0.010), columnspacing=1.8,
               handlelength=1.7, handletextpad=0.5)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig4.{ext}")
        fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
