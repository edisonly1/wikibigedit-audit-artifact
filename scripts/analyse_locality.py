"""Analyse the reconstructed locality experiment.

The contrast of interest is fuzzy vs random probe selection (matched on relation and
template). Since each item contributes all strata under all conditions, the data are
paired within item, so the tests are McNemar on the fuzzy/random pairs plus a
GEE clustered on item. Two controls should hold: the null condition shows no
fuzzy/random gap, and retrieval pulls the item's own edit far more often for fuzzy
probes than random ones.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.llm import normalise as normalise_local  # noqa: E402
from wbe_audit.stats import wilson  # noqa: E402

METHODS = ("oracle", "retrieval", "scope_gated", "null")
STRATA = ("fuzzy", "random", "same")

# Prepending any fact sometimes makes the model refuse rather than answer. That is
# a locality failure by the strict definition but it is not knowledge leakage, so
# the two are reported separately.
REFUSALS = {"unknown", "not known", "no answer", "none", "n a", "unclear",
            "not specified", "not available", "cannot determine"}


def mcnemar(b: int, c: int):
    """Exact two-sided McNemar on discordant pairs (b, c)."""
    from scipy import stats
    n = b + c
    if n == 0:
        return 1.0, 0, 0
    p = stats.binomtest(b, n, 0.5).pvalue
    return p, b, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--gee", action="store_true", help="also fit the GEE model")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.path, encoding="utf-8-sig")
            if l.strip()]
    print(f"{len(rows)} items from {args.path}")
    if not rows:
        return
    print(f"model: {rows[0].get('model')}\n")

    # ---------------- descriptive ----------------
    def cell(v):
        w = wilson(sum(v), len(v))
        return f"{100*w.point:5.1f} [{100*w.lo:4.1f},{100*w.hi:5.1f}]"

    print("=== locality by condition x stratum (% [95% CI]) ===")
    print(f"  {'condition':12s} " + "".join(f"{s:>22s}" for s in STRATA))
    loc = defaultdict(lambda: defaultdict(list))
    leak = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for s in STRATA:
            p = r["probes"][s]
            for m in METHODS:
                loc[m][s].append(bool(p["locality"][m]))
                leak[m][s].append(bool(p["leakage"][m]))
    for m in METHODS:
        print(f"  {m:12s} " + "".join(f"{cell(loc[m][s]):>22s}" for s in STRATA))

    print("\n=== leakage: answer contains the applied edit's object ===")
    print(f"  {'condition':12s} " + "".join(f"{s:>22s}" for s in STRATA))
    for m in METHODS:
        print(f"  {m:12s} " + "".join(f"{cell(leak[m][s]):>22s}" for s in STRATA))

    # ---------------- primary paired test ----------------
    print("\n=== PRIMARY: fuzzy vs random locality, paired within item ===")
    for m in METHODS:
        b = c = 0          # b: fuzzy fails & random holds; c: the reverse
        for r in rows:
            f = bool(r["probes"]["fuzzy"]["locality"][m])
            g = bool(r["probes"]["random"]["locality"][m])
            if not f and g:
                b += 1
            elif f and not g:
                c += 1
        p, b, c = mcnemar(b, c)
        nf = sum(loc[m]["fuzzy"])
        nr = sum(loc[m]["random"])
        n = len(rows)
        delta = (nf - nr) / n
        print(f"  {m:12s} fuzzy {100*nf/n:5.1f}%  random {100*nr/n:5.1f}%  "
              f"delta {100*delta:+5.1f}pp   discordant {b}/{c}   p = {p:.3g}")

    print("\n=== SECONDARY: fuzzy vs random LEAKAGE, paired within item ===")
    print("  (adoption of the applied edit's object -- a targeted measure, unlike")
    print("   binary locality which is saturated by refusals and drift)")
    for m in METHODS:
        b = c = 0      # b: fuzzy leaks & random does not; c: the reverse
        for r in rows:
            f = bool(r["probes"]["fuzzy"]["leakage"][m])
            g = bool(r["probes"]["random"]["leakage"][m])
            if f and not g:
                b += 1
            elif g and not f:
                c += 1
        p, b, c = mcnemar(b, c)
        nf = sum(leak[m]["fuzzy"])
        nr = sum(leak[m]["random"])
        n = len(rows)
        print(f"  {m:12s} fuzzy {100*nf/n:5.1f}%  random {100*nr/n:5.1f}%  "
              f"delta {100*(nf-nr)/n:+5.1f}pp   discordant {b}/{c}   p = {p:.3g}")

    print("\n=== dose-response: LEAKAGE vs probe-subject similarity ===")
    bins2 = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 1.01)]
    for m in ("oracle", "retrieval", "null"):
        print(f"  {m}:")
        for lo, hi in bins2:
            v = []
            for r in rows:
                for s in ("fuzzy", "random"):
                    p_ = r["probes"][s]
                    if lo <= p_["similarity"] < hi:
                        v.append(bool(p_["leakage"][m]))
            if len(v) >= 15:
                print(f"    sim [{lo:.2f},{hi:.2f})  n={len(v):4d}  "
                      f"leakage {wilson(sum(v), len(v)).pct()}")

    # ---- difference-in-differences against the null-edit condition ----
    # The fuzzy stratum may differ from random for reasons unrelated to the edit
    # (harder entities, less stable answers). The null condition measures exactly
    # that. The edit-attributable effect is therefore
    #     (fuzzy - random | real edit) - (fuzzy - random | null edit)
    # tested per item as a difference of differences.
    print("\n=== DIFFERENCE-IN-DIFFERENCES vs the null-edit control ===")
    print("  isolates the edit-attributable part of any fuzzy/random gap")
    try:
        from scipy import stats as _st
        for measure in ("locality", "leakage"):
            print(f"  --- {measure} ---")
            for m in ("oracle", "retrieval"):
                d = []
                for r in rows:
                    f_m = int(r["probes"]["fuzzy"][measure][m])
                    g_m = int(r["probes"]["random"][measure][m])
                    f_n = int(r["probes"]["fuzzy"][measure]["null"])
                    g_n = int(r["probes"]["random"][measure]["null"])
                    d.append((f_m - g_m) - (f_n - g_n))
                mean_d = sum(d) / len(d)
                nz = [x for x in d if x != 0]
                if nz:
                    w, p = _st.wilcoxon(d, zero_method="wilcox",
                                        alternative="two-sided")
                else:
                    p = 1.0
                lo, hi = _st.bootstrap((d,), lambda x, axis=None: x.mean(axis=axis),
                                       n_resamples=4000, random_state=0
                                       ).confidence_interval
                print(f"    {m:11s} DiD {100*mean_d:+6.2f}pp "
                      f"[{100*lo:+.2f}, {100*hi:+.2f}]  "
                      f"nonzero {len(nz)}/{len(d)}  p = {p:.3g}")
    except Exception as exc:
        print(f"    (unavailable: {type(exc).__name__}: {exc})")

    print("\n=== same-subject stratum vs fuzzy (secondary, not relation-matched) ===")
    for m in METHODS:
        b = c = 0
        for r in rows:
            f = bool(r["probes"]["fuzzy"]["locality"][m])
            g = bool(r["probes"]["same"]["locality"][m])
            if not f and g:
                b += 1
            elif f and not g:
                c += 1
        p, b, c = mcnemar(b, c)
        print(f"  {m:12s} discordant {b}/{c}  p = {p:.3g}")

    # ---------------- controls ----------------
    print("\n=== CONTROL 1: null edit must show no fuzzy/random gap ===")
    nf = sum(loc["null"]["fuzzy"])
    nr = sum(loc["null"]["random"])
    n = len(rows)
    b = sum(1 for r in rows
            if not r["probes"]["fuzzy"]["locality"]["null"]
            and r["probes"]["random"]["locality"]["null"])
    c = sum(1 for r in rows
            if r["probes"]["fuzzy"]["locality"]["null"]
            and not r["probes"]["random"]["locality"]["null"])
    p, _, _ = mcnemar(b, c)
    print(f"  null: fuzzy {100*nf/n:.1f}%  random {100*nr/n:.1f}%  p = {p:.3g}")
    print("  (a significant gap here would mean the effect is not edit-driven)")

    print("\n=== CONTROL 2: retrieval pulls the item's own edit ===")
    for s in STRATA:
        own = [bool(r["probes"][s]["retrieved_is_own"]) for r in rows]
        sims = [r["probes"][s]["retrieval_sim"] for r in rows]
        print(f"  {s:8s} own-edit retrieved {wilson(sum(own), len(own)).pct()}   "
              f"mean top-1 sim {sum(sims)/len(sims):.3f}")

    print("\n=== CONTROL 3: scope gate fires ===")
    for s in STRATA:
        g = [bool(r["probes"][s]["scope_gate_yes"]) for r in rows]
        print(f"  {s:8s} gate=YES {wilson(sum(g), len(g)).pct()}")

    # ---------------- decomposition of locality failures ----------------
    print("\n=== what kind of locality failure? (share of all probes) ===")
    print(f"{'condition':13s} {'stratum':8s} {'leaked':>10s} {'refused':>10s} "
          f"{'other change':>14s} {'held':>10s}")
    for m in METHODS:
        for s in STRATA:
            leaked = refused = other = held = 0
            for r in rows:
                p = r["probes"][s]
                if p["locality"][m]:
                    held += 1
                    continue
                ans = normalise_local(p["answers"][m])
                if p["leakage"][m]:
                    leaked += 1
                elif ans in REFUSALS or "unknown" in ans:
                    refused += 1
                else:
                    other += 1
            n = len(rows)
            print(f"  {m:11s} {s:8s} {100*leaked/n:9.1f}% {100*refused/n:9.1f}% "
                  f"{100*other/n:13.1f}% {100*held/n:9.1f}%")

    # ---------------- dose-response on similarity ----------------
    print("\n=== dose-response: locality vs probe-subject similarity ===")
    bins = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 1.01)]
    for m in ("oracle", "retrieval", "null"):
        print(f"  {m}:")
        for lo, hi in bins:
            v = []
            for r in rows:
                for s in ("fuzzy", "random"):
                    p = r["probes"][s]
                    if lo <= p["similarity"] < hi:
                        v.append(bool(p["locality"][m]))
            if len(v) >= 15:
                print(f"    sim [{lo:.2f},{hi:.2f})  n={len(v):4d}  "
                      f"locality {wilson(sum(v), len(v)).pct()}")

    # ---------------- is locality measurable on this item set at all? ----------
    print("\n=== does the model know the probe answers? ===")
    print("  (WikiBigEdit scores locality as 'still produces the gold answer'. If")
    print("   baseline accuracy is at the floor, that metric cannot move.)")
    from wbe_audit.llm import contains_answer as _contains
    for s in STRATA:
        base_correct = [_contains(r["probes"][s]["baseline"], r["probes"][s]["gold"])
                        for r in rows]
        print(f"  {s:8s} baseline answer matches gold: "
              f"{wilson(sum(base_correct), len(base_correct)).pct()}")

    print("\n=== WikiBigEdit-style locality (post-edit answer still matches gold) ===")
    print(f"  {'condition':12s} " + "".join(f"{s:>22s}" for s in STRATA))
    for m in METHODS:
        cells = []
        for s in STRATA:
            v = [_contains(r["probes"][s]["answers"][m], r["probes"][s]["gold"])
                 for r in rows]
            cells.append(cell(v))
        print(f"  {m:12s} " + "".join(f"{c:>22s}" for c in cells))

    print("\n=== edit success (oracle condition) ===")
    es = [bool(r["edit_success"]["oracle"]) for r in rows]
    print(f"  counterfactual asserted in answer: {wilson(sum(es), len(es)).pct()}")

    print("\n=== probe subject-label similarity (manipulation check) ===")
    for s in STRATA:
        v = [r["probes"][s]["similarity"] for r in rows]
        print(f"  {s:8s} mean {sum(v)/len(v):.3f}")

    # ---------------- model-based interaction ----------------
    if args.gee:
        try:
            import numpy as np
            import pandas as pd
            import statsmodels.api as sm
            import statsmodels.formula.api as smf

            recs = []
            for i, r in enumerate(rows):
                for s in STRATA:
                    for m in METHODS:
                        recs.append({"item": i, "prop": r["edit_property_id"],
                                     "stratum": s, "method": m,
                                     "loc": int(r["probes"][s]["locality"][m])})
            df = pd.DataFrame(recs)
            df = df[df["stratum"].isin(["fuzzy", "random"])]
            df["stratum"] = pd.Categorical(df["stratum"],
                                           categories=["random", "fuzzy"])
            df["method"] = pd.Categorical(df["method"],
                                          categories=["null", "oracle",
                                                      "retrieval", "scope_gated"])
            mod = smf.gee("loc ~ C(method) * C(stratum)", groups="item", data=df,
                          family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable())
            res = mod.fit()
            print("\n=== GEE: loc ~ method * stratum, clustered on item ===")
            print(res.summary().tables[1])
        except Exception as exc:
            print(f"\n(GEE unavailable: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
