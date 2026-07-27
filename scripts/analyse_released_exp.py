"""Analysis of the released-probe experiment.

Contrast: WikiBigEdit's own `loc` probe for a row (fuzzy-selected against that
row's edit subject by construction) versus another row's `loc` probe (same probe
population, not similarity-matched). Difference-in-differences against an
unrelated-edit control isolates the edit-attributable part.

Reports WikiBigEdit's own locality definition ("still produces loc_ans") alongside
the answer-unchanged definition, since the former is what the benchmark scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scipy import stats  # noqa: E402

from wbe_audit.stats import wilson  # noqa: E402


def mcnemar(b, c):
    n = b + c
    return (stats.binomtest(b, n, 0.5).pvalue if n else 1.0), b, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path, encoding="utf-8-sig")
            if l.strip()]
    n = len(rows)
    print(f"{n} items from {args.path}\n")

    print("=== pre-edit accuracy on the released loc probes ===")
    for k in ("own", "other"):
        v = [r["probes"][k]["baseline_correct"] for r in rows]
        print(f"  {k:6s} {wilson(sum(v), len(v)).pct()}")

    for measure, label in (("locality", "locality (answer unchanged)"),
                           ("gold_kept", "WikiBigEdit locality (still gives loc_ans)"),
                           ("leakage", "leakage (answer contains the edit's object)")):
        print(f"\n=== {label} ===")
        for cond in ("edit", "null"):
            cells = []
            for k in ("own", "other"):
                v = [r["probes"][k][measure][cond] for r in rows]
                cells.append(f"{k} {wilson(sum(v), len(v)).pct()}")
            print(f"  {cond:5s}  " + "   ".join(cells))

        # paired own-vs-other within each condition
        for cond in ("edit", "null"):
            b = sum(1 for r in rows
                    if r["probes"]["own"][measure][cond]
                    and not r["probes"]["other"][measure][cond])
            c = sum(1 for r in rows
                    if not r["probes"]["own"][measure][cond]
                    and r["probes"]["other"][measure][cond])
            p, b, c = mcnemar(b, c)
            print(f"    paired own vs other [{cond}]: discordant {b}/{c}  p = {p:.3g}")

        # difference-in-differences
        d = []
        for r in rows:
            o_e = int(r["probes"]["own"][measure]["edit"])
            t_e = int(r["probes"]["other"][measure]["edit"])
            o_n = int(r["probes"]["own"][measure]["null"])
            t_n = int(r["probes"]["other"][measure]["null"])
            d.append((o_e - t_e) - (o_n - t_n))
        mean_d = sum(d) / len(d)
        nz = [x for x in d if x]
        p = stats.wilcoxon(d).pvalue if nz else 1.0
        lo, hi = stats.bootstrap((d,), lambda x, axis=None: x.mean(axis=axis),
                                 n_resamples=4000, random_state=0).confidence_interval
        print(f"    DiD {100*mean_d:+6.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}]  "
              f"nonzero {len(nz)}/{len(d)}  p = {p:.3g}")


if __name__ == "__main__":
    main()
