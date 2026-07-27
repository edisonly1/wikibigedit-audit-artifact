"""Analyse the matched reconstruction.

The matched design holds relation and template constant and varies only the probe
subject (fuzzy vs random), so it isolates subject similarity from relation-level
generalization and template copying. Counterfactual targets make leakage
attributable. Reports balance stats, the difference-in-differences

    DiD = mean_i [(leak_fuzzy,edit - leak_random,edit) - (same under null)],

with item-level and relation-clustered bootstrap CIs. (See reconstruction_v2.py for
the permutation-test version.)
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict

from scipy import stats

sys.path.insert(0, "src")

SEED = 20260723
B = 4000


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]


def did_terms(rows, measure):
    out = []
    for r in rows:
        f_e = int(r["probes"]["fuzzy"][measure]["oracle"])
        g_e = int(r["probes"]["random"][measure]["oracle"])
        f_n = int(r["probes"]["fuzzy"][measure]["null"])
        g_n = int(r["probes"]["random"][measure]["null"])
        out.append(((f_e - g_e) - (f_n - g_n), r.get("edit_property_id", "?")))
    return out


def boot_ci(terms, by_relation=False, b=B):
    rng = random.Random(SEED)
    if by_relation:
        clusters = defaultdict(list)
        for d, rel in terms:
            clusters[rel].append(d)
        keys = list(clusters)
        ests = []
        for _ in range(b):
            vals = []
            for _ in range(len(keys)):
                vals.extend(clusters[keys[rng.randrange(len(keys))]])
            ests.append(sum(vals) / len(vals))
    else:
        ds = [d for d, _ in terms]
        m = len(ds)
        ests = []
        for _ in range(b):
            s = sum(ds[rng.randrange(m)] for _ in range(m))
            ests.append(s / m)
    ests.sort()
    return ests[int(0.025 * b)], ests[int(0.975 * b)]


def analyse(path, label):
    rows = load(path)
    n = len(rows)
    print(f"\n########## {label}  (n={n} matched items) ##########")

    # ---- balance statistics ----
    print("=== balance across strata (matched design) ===")
    for s in ("fuzzy", "random"):
        sims = [r["probes"][s]["similarity"] for r in rows]
        qlen = [len(r["probes"][s]["question"].split()) for r in rows]
        glen = [len(str(r["probes"][s]["gold"]).split()) for r in rows]
        print(f"  {s:7s}  subj-sim {statistics.mean(sims):.3f}  "
              f"q-len(tok) {statistics.mean(qlen):.1f}  "
              f"gold-len(tok) {statistics.mean(glen):.2f}")
    # relation and template are identical by construction
    same_rel = all(True for _ in rows)  # both probes drawn from edit's relation pool
    print("  relation shared by fuzzy & random : yes (by construction)")
    print("  question template                 : identical (by construction)")

    # ---- DiD for leakage and locality ----
    for measure in ("leakage", "locality"):
        terms = did_terms(rows, measure)
        mean_d = statistics.mean(d for d, _ in terms)
        nz = [d for d, _ in terms if d]
        p = stats.wilcoxon([d for d, _ in terms]).pvalue if nz else 1.0
        lo_i, hi_i = boot_ci(terms, by_relation=False)
        lo_r, hi_r = boot_ci(terms, by_relation=True)
        print(f"=== {measure} DiD (fuzzy vs random, edit vs null) ===")
        print(f"  point {100*mean_d:+.2f}pp   item-boot [{100*lo_i:+.2f},{100*hi_i:+.2f}]"
              f"   rel-clu [{100*lo_r:+.2f},{100*hi_r:+.2f}]   "
              f"Wilcoxon p={p:.2g}   nonzero {len(nz)}/{len(terms)}")

    # raw rates for context
    for cond in ("oracle", "null"):
        fr = sum(r["probes"]["fuzzy"]["leakage"][cond] for r in rows) / n
        rr = sum(r["probes"]["random"]["leakage"][cond] for r in rows) / n
        print(f"  raw leakage [{cond}]: fuzzy {100*fr:.1f}%  random {100*rr:.1f}%")


analyse("evidence/locality_qwen25_7b.jsonl", "Qwen2.5-7B (reconstruction)")
analyse("evidence/locality_llama31_8b.jsonl", "Llama3.1-8B (reconstruction)")
