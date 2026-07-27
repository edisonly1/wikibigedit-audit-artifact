"""Leakage/locality effect conditional on the edit actually succeeding.

A failed weight edit cannot leak, so the all-rows DiD understates the effect on
edits that took. This recomputes the own-vs-other, edit-vs-null DiD on the
successful-edit subgroup.
"""
import json
import sys

from scipy import stats

sys.path.insert(0, "src")
from wbe_audit.stats import wilson  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1
        else "evidence/param_exp_ftl.jsonl", encoding="utf-8-sig") if l.strip()]
succ = [r for r in rows if r.get("edit_success")]
print(f"{len(rows)} rows, {len(succ)} successful edits\n")

for name, subset in (("all rows", rows), ("successful edits", succ)):
    print(f"=== {name} (n={len(subset)}) ===")
    for measure in ("leakage", "locality", "gold_kept"):
        d = []
        oe = ot = 0
        for r in subset:
            o_e = int(r["probes"]["own"][measure]["edit"])
            t_e = int(r["probes"]["other"][measure]["edit"])
            o_n = int(r["probes"]["own"][measure]["null"])
            t_n = int(r["probes"]["other"][measure]["null"])
            d.append((o_e - t_e) - (o_n - t_n))
            oe += o_e
            ot += t_e
        mean_d = sum(d) / len(d)
        nz = [x for x in d if x]
        p = stats.wilcoxon(d).pvalue if nz else 1.0
        lo, hi = stats.bootstrap((d,), lambda x, axis=None: x.mean(axis=axis),
                                 n_resamples=4000, random_state=0).confidence_interval
        tag = {"gold_kept": "WikiBigEdit metric"}.get(measure, measure)
        print(f"  {tag:20s} own {100*oe/len(subset):5.1f}%  other "
              f"{100*ot/len(subset):5.1f}%  DiD {100*mean_d:+6.2f}pp "
              f"[{100*lo:+.2f},{100*hi:+.2f}]  p={p:.3g}")
    print()
