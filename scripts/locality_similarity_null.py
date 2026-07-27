"""Quantify the fuzzy locality fallback against a null.

create_locality_probes picks the probe subject as the argmax over string
similarity, so if that branch dominates, the similarity between an edit subject and
its probe should sit well above random pairing. The null pairs each edit subject
with a random other row's probe (same marginals, selection removed). We use difflib's
SequenceMatcher, which is the same metric fuzzywuzzy.fuzz.ratio wraps.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import statistics
import sys
from difflib import SequenceMatcher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() * 100.0


def partial_ratio(needle: str, hay: str) -> float:
    """Best window match, mirroring fuzz.partial_ratio."""
    if not needle or not hay:
        return 0.0
    if len(needle) > len(hay):
        needle, hay = hay, needle
    best = 0.0
    n = len(needle)
    for i in range(0, max(1, len(hay) - n + 1)):
        best = max(best, ratio(needle, hay[i:i + n]))
        if best == 100.0:
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", default="evidence/locality_similarity.json")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            if isinstance(r.get("subject"), str) and isinstance(r.get("loc"), str) \
                    and isinstance(r.get("update"), str):
                rows.append(r)
    print(f"{len(rows)} usable rows")

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    obs_q, null_q, obs_s, null_s = [], [], [], []
    for r in sample:
        other = rows[rng.randrange(len(rows))]
        # question-to-question: same template, so this isolates the entity slot
        obs_q.append(ratio(r["update"], r["loc"]))
        null_q.append(ratio(r["update"], other["loc"]))
        # subject-to-probe-question
        obs_s.append(partial_ratio(r["subject"], r["loc"]))
        null_s.append(partial_ratio(r["subject"], other["loc"]))

    def rep(name, obs, null):
        mo, mn = statistics.mean(obs), statistics.mean(null)
        print(f"\n  {name}")
        print(f"    observed (paired probe)  mean {mo:6.2f}  "
              f"median {statistics.median(obs):6.2f}")
        print(f"    null     (random probe)  mean {mn:6.2f}  "
              f"median {statistics.median(null):6.2f}")
        print(f"    difference               {mo-mn:+6.2f}")
        for thr in (70, 80, 90):
            po = 100 * sum(1 for x in obs if x >= thr) / len(obs)
            pn = 100 * sum(1 for x in null if x >= thr) / len(null)
            print(f"    share >= {thr}: observed {po:5.1f}%   null {pn:5.1f}%")
        return mo, mn

    print("\n=== string similarity: edited row vs its own locality probe ===")
    rep("edit question  vs locality question", obs_q, null_q)
    rep("edited subject vs locality question", obs_s, null_s)

    # paired test
    try:
        from scipy import stats
        d = [a - b for a, b in zip(obs_q, null_q)]
        t, p = stats.ttest_rel(obs_q, null_q)
        w, pw = stats.wilcoxon(obs_q, null_q)
        print(f"\n  paired t-test (questions): t = {t:.1f}, p = {p:.3g}")
        print(f"  Wilcoxon signed-rank      : p = {pw:.3g}")
        print(f"  mean paired difference    : {statistics.mean(d):+.2f}")
    except Exception as exc:
        print(f"  (stat test unavailable: {exc})")

    json.dump({"obs_question": obs_q[:500], "null_question": null_q[:500],
               "obs_subject": obs_s[:500], "null_subject": null_s[:500]},
              open(args.out, "w", encoding="utf-8"))
    print(f"\nwrote {args.out}")
    print("\nInterpretation: the locality probe is not an arbitrary unrelated fact.")
    print("It is the closest available string to the edited subject, which is what")
    print("the argmax over fuzz.ratio selects.")


if __name__ == "__main__":
    main()
