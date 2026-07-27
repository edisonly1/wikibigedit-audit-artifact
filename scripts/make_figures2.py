"""Additional figures for the expanded paper: cross-paradigm leakage and the
leakage dose-response over probe similarity."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wbe_audit.stats import wilson  # noqa: E402

os.makedirs("paper/figs", exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, PURPLE = ("#3b6ea5", "#e08e45", "#4a9c6d", "#c0504d",
                                    "#7a5c99")


# ---- Figure 4: cross-paradigm leakage DiD with 95% bootstrap CIs -------------
# (point, lo, hi) from analyse_released_exp.py / param_conditional.py
leak = [
    ("in-context/retr.\n(Qwen-7B)", 33.75, 29.00, 39.00, BLUE),
    ("in-context/retr.\n(Llama-8B)", 23.75, 19.50, 28.25, BLUE),
    ("param FT-L\n(Qwen-1.5B)", 22.33, 17.30, 27.67, ORANGE),
    ("param ROME\n(Qwen-1.5B)", 20.33, 15.67, 25.33, ORANGE),
]
fig, ax = plt.subplots(figsize=(4.3, 2.9))
xs = np.arange(len(leak))
for i, (lab, p, lo, hi, c) in enumerate(leak):
    ax.bar(i, p, color=c, alpha=0.85, width=0.62)
    ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="k", lw=1.1, capsize=3)
    ax.text(i, hi + 1.0, f"{p:.0f}", ha="center", fontsize=8.5)
ax.set_xticks(xs)
ax.set_xticklabels([l[0] for l in leak], fontsize=7.5)
ax.set_ylabel("leakage DiD (pp)")
ax.set_ylim(0, 44)
ax.axhline(0, color="k", lw=0.6)
ax.set_title("Edit-to-neighbour leakage replicates across paradigms")
fig.tight_layout()
fig.savefig("paper/figs/fig4_leakage_bars.pdf")
plt.close(fig)


# ---- Figure 5: leakage dose-response over probe-subject similarity -----------
rows = [json.loads(l) for l in open("evidence/locality_qwen25_7b.jsonl",
                                    encoding="utf-8-sig") if l.strip()]
bins = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 1.01)]
centres = [0.15, 0.375, 0.525, 0.675, 0.88]


def curve(measure_key):
    pts, los, his = [], [], []
    for lo, hi in bins:
        v = []
        for r in rows:
            for s in ("fuzzy", "random"):
                p = r["probes"][s]
                if lo <= p["similarity"] < hi:
                    v.append(bool(p["leakage"][measure_key]))
        w = wilson(sum(v), len(v))
        pts.append(100 * w.point); los.append(100 * w.lo); his.append(100 * w.hi)
    return pts, los, his


fig, ax = plt.subplots(figsize=(4.3, 2.9))
for key, c, lab in [("oracle", RED, "real edit applied"),
                    ("null", GREEN, "unrelated edit (control)")]:
    p, lo, hi = curve(key)
    ax.plot(centres, p, "o-", color=c, label=lab)
    ax.fill_between(centres, lo, hi, color=c, alpha=0.15)
ax.set_xlabel("probe-subject string similarity to the edit subject")
ax.set_ylabel("leakage (%)")
ax.set_title("Leakage rises with lexical similarity")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_ylim(0, 45)
fig.tight_layout()
fig.savefig("paper/figs/fig5_doseresponse.pdf")
plt.close(fig)

print("wrote paper/figs/fig4_leakage_bars.pdf, fig5_doseresponse.pdf")
