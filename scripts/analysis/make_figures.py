"""Build README figures from evaluation JSONs in this repo (no NSCC access needed).

Usage: python scripts/analysis/make_figures.py
"""
import glob
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(ROOT, "results/training")
AUTHOR = os.path.join(ROOT, "results/author_released/evaluation.json")
OUT = os.path.join(ROOT, "docs/assets")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
AUTHOR_REPORTED = 0.8528

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 11,
})

# ---------- training curve (two small multiples, no dual axis) ----------
rows = []
for f in sorted(glob.glob(os.path.join(TRAIN, "evaluation_0*.json"))):
    d = json.load(open(f))
    rows.append((d["iteration"], d["successes"], d["normalized_return"]))
it = [r[0] / 1000 for r in rows]
succ = [r[1] for r in rows]
ret = [r[2] for r in rows]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
a1.plot(it, ret, color=BLUE, lw=2, marker="o", ms=5, zorder=3)
a1.axhline(AUTHOR_REPORTED, color=INK2, lw=1.2, ls="--", zorder=2)
a1.text(0.2, AUTHOR_REPORTED + 0.012, f"author's released policy: {AUTHOR_REPORTED}", color=INK2, fontsize=9)
a1.annotate(f"ours: {ret[-1]:.3f}", (it[-1], ret[-1]), xytext=(-70, -28), textcoords="offset points",
            color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a1.set_title("Normalized return (40 LAFAN1 clips, full-clip eval)", loc="left", fontsize=11, color=INK)
a1.set_xlabel("training iteration (×1000)")
a1.set_ylim(0, 1.0)

a2.plot(it, succ, color=BLUE, lw=2, marker="o", ms=5, zorder=3)
a2.axhline(40, color=INK2, lw=1.2, ls="--", zorder=2)
a2.set_title("Clips tracked to the end without falling", loc="left", fontsize=11, color=INK)
a2.set_xlabel("training iteration (×1000)")
a2.set_ylim(0, 42)
a2.set_yticks([0, 10, 20, 30, 40])
fig.suptitle("SuperMimic on Unitree G1 — our training run (1× A100 40GB, 6.5 h)", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "training_curve.png"))
plt.close(fig)

# ---------- per-category comparison (grouped bars) ----------
mine = {m["motion"]: m for m in json.load(open(os.path.join(TRAIN, "evaluation_015000.json")))["motions"]}
auth = {m["motion"]: m for m in json.load(open(AUTHOR))["motions"]}
cats = {}
for n in mine:
    c = n.split("_")[0].rstrip("0123456789")
    cats.setdefault(c, []).append(n)
order = sorted(cats, key=lambda c: st.mean(auth[n]["normalized_return"] for n in cats[c]))
labels = [f"{c}\n({len(cats[c])})" for c in order]
va = [st.mean(auth[n]["normalized_return"] for n in cats[c]) for c in order]
vm = [st.mean(mine[n]["normalized_return"] for n in cats[c]) for c in order]

# Dot plot (not bars): the interesting range is 0.78-0.92, and bars must start at zero.
fig, ax = plt.subplots(figsize=(11, 3.8), dpi=150)
x = list(range(len(order)))
for i in x:
    ax.plot([i, i], [va[i], vm[i]], color=GRID, lw=3, zorder=2, solid_capstyle="round")
ax.scatter(x, va, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="author's released weights (evaluated by us)")
ax.scatter(x, vm, s=70, color=ORANGE, marker="D", edgecolor=SURFACE, linewidth=2, zorder=4, label="our training run")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0.78, 0.92)
ax.set_ylabel("normalized return")
ax.set_title("Per motion category, mean over clips (number of clips in brackets)", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_category.png"))
plt.close(fig)
print("wrote", os.listdir(OUT))
