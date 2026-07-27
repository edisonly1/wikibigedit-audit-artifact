"""Cluster-robust prevalence intervals and the sampling inclusion flow.

Rows aren't independent: they share subjects, relations, intervals and source
revisions, so row-level Wilson intervals can be too narrow. This recomputes each
headline prevalence with a cluster bootstrap (resampling whole clusters) for three
clustering choices, reports the design effect against Wilson, and prints the
inclusion flow (sampled -> resolvable -> classified, with unresolvable reasons).
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from wbe_audit.stats import wilson  # noqa: E402

RECLASS = "evidence/hist_n1500.reclassified.jsonl"
RAW_AUDIT = "evidence/hist_n1500.jsonl"   # before object reclassification
SEED = 20260722
B = 4000

INVALID = {"role-erased-qualifier", "role-erased-reference",
           "property-absent-on-subject", "object-qid-misattached",
           "object-genuinely-wrong", "object-absent-unresolved"}
GROUPS = {
    "invalid (rho_WBE)": lambda c: c in INVALID,
    "endpoint-valid": lambda c: c == "endpoint-valid",
    "role erasure": lambda c: c in {"role-erased-qualifier", "role-erased-reference"},
    "identity (property absent)": lambda c: c == "property-absent-on-subject",
    "object-QID misattached": lambda c: c == "object-qid-misattached",
    "text supervision defensible": lambda c: c in {
        "endpoint-valid", "main-supported-nonunique", "object-qid-misattached"},
    "structured metadata correct": lambda c: c in {
        "endpoint-valid", "main-supported-nonunique"},
}


def interval_of(audit_id: str) -> str:
    m = re.search(r"(\d{8}_\d{8})", audit_id)
    return m.group(1) if m else "?"


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]


def cluster_boot(rows, keyfn, predicate, b=B, seed=SEED):
    """Percentile CI for the mean of predicate, resampling clusters."""
    clusters = defaultdict(list)
    for r in rows:
        clusters[keyfn(r)].append(bool(predicate(r["operational_class"])))
    keys = list(clusters)
    rng = random.Random(seed)
    n_total = sum(len(v) for v in clusters.values())
    ests = []
    for _ in range(b):
        num = den = 0
        for _ in range(len(keys)):
            c = clusters[keys[rng.randrange(len(keys))]]
            num += sum(c)
            den += len(c)
        ests.append(num / den if den else 0.0)
    ests.sort()
    lo = ests[int(0.025 * b)]
    hi = ests[int(0.975 * b)]
    return lo, hi, len(keys), n_total


def main() -> None:
    # ---- inclusion flow ----
    raw = load(RAW_AUDIT)
    n_sampled = len(raw)
    unresolved = [r for r in raw if r.get("operational_class") == "subject-unavailable"
                  or not r.get("operational_class")]
    reasons = Counter(r.get("error") or "unknown" for r in unresolved)

    rows = [r for r in load(RECLASS)
            if r.get("operational_class")
            and r["operational_class"] != "subject-unavailable"]
    n = len(rows)

    print("=== sampling inclusion flow ===")
    print(f"  population (public release rows)      : 502,382")
    print(f"  design                                : uniform SRS over rows, "
          f"seed {SEED}, no weights")
    print(f"  rows drawn                            : {n_sampled}")
    print(f"  unresolvable (subject/revision)       : {len(unresolved)}")
    for k, v in reasons.most_common(6):
        print(f"      {k}: {v}")
    print(f"  resolvable & classified               : {n}")

    # cluster structure of the sample
    subj = Counter(r["subject_id"] for r in rows)
    rel = Counter(r["property_id"] for r in rows)
    itv = Counter(interval_of(r["audit_id"]) for r in rows)
    print("\n=== cluster structure of the sample ===")
    print(f"  distinct subjects  : {len(subj)}  (max rows/subject {max(subj.values())}, "
          f"share of subjects with >1 row {100*sum(1 for v in subj.values() if v>1)/len(subj):.1f}%)")
    print(f"  distinct relations : {len(rel)}  (max rows/relation {max(rel.values())}, "
          f"top relation {100*max(rel.values())/n:.1f}% of rows)")
    print(f"  distinct intervals : {len(itv)}")

    keyfns = {
        "subject": lambda r: r["subject_id"],
        "relation": lambda r: r["property_id"],
        "interval": lambda r: interval_of(r["audit_id"]),
    }

    print("\n=== prevalence with row-level Wilson vs cluster bootstrap CIs ===")
    hdr = f"{'quantity':30s} {'point':>6s} {'Wilson':>16s}"
    for cl in keyfns:
        hdr += f" {'clu:'+cl:>16s}"
    print(hdr)
    for name, pred in GROUPS.items():
        k = sum(1 for r in rows if pred(r["operational_class"]))
        w = wilson(k, n)
        line = f"{name:30s} {100*w.point:5.1f}  [{100*w.lo:4.1f},{100*w.hi:5.1f}]"
        wilson_width = w.hi - w.lo
        widths = {}
        for cl, kf in keyfns.items():
            lo, hi, nk, _ = cluster_boot(rows, kf, pred)
            line += f"  [{100*lo:4.1f},{100*hi:5.1f}]"
            widths[cl] = hi - lo
        print(line)
    # design effects on the headline number
    print("\n=== design effect (cluster CI width / Wilson width)^1 for 'invalid' ===")
    pred = GROUPS["invalid (rho_WBE)"]
    k = sum(1 for r in rows if pred(r["operational_class"]))
    w = wilson(k, n)
    for cl, kf in keyfns.items():
        lo, hi, nk, _ = cluster_boot(rows, kf, pred)
        deff = ((hi - lo) / (w.hi - w.lo)) ** 2
        print(f"  cluster={cl:9s} width ratio {(hi-lo)/(w.hi-w.lo):.2f}  "
              f"approx design effect {deff:.2f}  ({nk} clusters)")


if __name__ == "__main__":
    main()
