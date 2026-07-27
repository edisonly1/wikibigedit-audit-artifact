"""Generate the paper's figures and LaTeX tables from the evidence files."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wbe_audit.stats import wilson  # noqa: E402

os.makedirs("paper/figs", exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})

BLUE, ORANGE, GREEN, RED, GREY = "#3b6ea5", "#e08e45", "#4a9c6d", "#c0504d", "#888"


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]


# ---- Figure 1: operational-class prevalence with Wilson CIs -------------------
rows = [r for r in load("evidence/hist_n1500.reclassified.jsonl")
        if r.get("operational_class")
        and r["operational_class"] != "subject-unavailable"]
n = len(rows)
order = ["endpoint-valid", "object-qid-misattached", "property-absent-on-subject",
         "role-erased-qualifier", "main-supported-nonunique",
         "object-absent-unresolved", "object-genuinely-wrong",
         "role-erased-reference"]
labels = {"endpoint-valid": "endpoint-valid", "object-qid-misattached": "obj-QID misattached",
          "property-absent-on-subject": "property absent (identity)",
          "role-erased-qualifier": "role-erased (qualifier)",
          "main-supported-nonunique": "main, non-unique",
          "object-absent-unresolved": "object absent (unresolved)",
          "object-genuinely-wrong": "object genuinely wrong",
          "role-erased-reference": "role-erased (reference)"}
col = {"endpoint-valid": GREEN, "main-supported-nonunique": GREEN,
       "object-qid-misattached": ORANGE, "object-absent-unresolved": ORANGE,
       "object-genuinely-wrong": RED, "property-absent-on-subject": RED,
       "role-erased-qualifier": BLUE, "role-erased-reference": BLUE}
from collections import Counter
c = Counter(r["operational_class"] for r in rows)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ys = range(len(order))
for i, k in enumerate(order):
    w = wilson(c[k], n)
    ax.barh(i, 100 * w.point, color=col[k], alpha=0.85, zorder=2)
    ax.plot([100 * w.lo, 100 * w.hi], [i, i], color="0.25", lw=1, zorder=3)
    # cap ticks
    for x in (100 * w.lo, 100 * w.hi):
        ax.plot([x, x], [i - 0.16, i + 0.16], color="0.25", lw=1, zorder=3)
    # label placed clear of the upper error-bar cap
    ax.text(100 * w.hi + 1.2, i, f"{100*w.point:.1f}%", va="center",
            ha="left", fontsize=8, zorder=4)
ax.set_yticks(list(ys))
ax.set_yticklabels([labels[k] for k in order])
ax.invert_yaxis()
ax.set_xlim(0, 62)
ax.set_xlabel(f"share of audited rows (%);  n = {n}, 95% Wilson CI")
ax.set_title("Operational-class prevalence at the interval-end snapshot")
fig.tight_layout()
fig.savefig("paper/figs/fig1_prevalence.pdf")
plt.close(fig)

# ---- Figure 2: timestamp sensitivity (validity plateaus) ---------------------
near = json.load(open("evidence/timestamp_calibration.json", encoding="utf-8"))
far = json.load(open("evidence/timestamp_calibration_far.json", encoding="utf-8"))


def valid_at(recs, d):
    av = [r for r in recs if r["t1"][str(d)].get("available")]
    k = sum(1 for r in av if r["t1"][str(d)]["primary_valid"])
    return 100 * k / len(av) if av else None


offs = [-14, -7, 0, 7, 14, 28, 56]
vals = []
for d in offs:
    v = valid_at(near, d) if d in (-14, -7, 0, 7) else valid_at(far, d)
    vals.append(v)
fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.plot(offs, vals, "o-", color=BLUE)
ax.axhline(vals[offs.index(0)], color=GREY, ls="--", lw=0.8)
ax.set_xlabel("snapshot offset from named dump date (days)")
ax.set_ylabel(r"$\rho_{\mathrm{WBE}}$ valid (%)")
ax.set_title("Validity plateaus; not a snapshot artifact")
ax.set_ylim(0, 60)
fig.tight_layout()
fig.savefig("paper/figs/fig2_timestamp.pdf")
plt.close(fig)

# ---- Figure 3: locality-probe entity provenance ------------------------------
lp = json.load(open("evidence/locality_provenance.json", encoding="utf-8"))["counts"]
tot = lp["usable"]
cats = [("same entity", lp["subject-string-in-probe"] + lp["all-subject-tokens-in-probe"]),
        ("partial overlap", lp["partial-token-overlap"]),
        ("different entity", lp["NO-overlap-different-entity"])]
fig, ax = plt.subplots(figsize=(4.2, 2.6))
left = 0
cols = [GREEN, ORANGE, RED]
for (lab, v), cc in zip(cats, cols):
    ax.barh(0, 100 * v / tot, left=left, color=cc, label=f"{lab} ({100*v/tot:.1f}%)")
    left += 100 * v / tot
ax.set_yticks([])
ax.set_xlim(0, 100)
ax.set_xlabel("share of 502,382 released locality probes (%)")
ax.set_title("Locality probes rarely concern the edited entity")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False,
          fontsize=7.5)
fig.tight_layout()
fig.savefig("paper/figs/fig3_locality_provenance.pdf")
plt.close(fig)

print("wrote paper/figs/fig1_prevalence.pdf, fig2_timestamp.pdf, "
      "fig3_locality_provenance.pdf")
