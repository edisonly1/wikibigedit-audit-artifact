"""Per-interval robustness and the unconditional decomposition.

Reports the per-interval invalid rate and the leave-one-interval-out range (to show
no single interval dominates), and the decomposition over all 1500 sampled rows with
an "unresolvable release record" fifth class (mostly rows with no subject
identifier, which are themselves defects).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from wbe_audit.stats import wilson  # noqa: E402

RECLASS = "evidence/hist_n1500.reclassified.jsonl"
RAW = "evidence/hist_n1500.jsonl"

INVALID = {"role-erased-qualifier", "role-erased-reference",
           "property-absent-on-subject", "object-qid-misattached",
           "object-genuinely-wrong", "object-absent-unresolved"}
UNUSABLE = {"role-erased-qualifier", "role-erased-reference",
            "property-absent-on-subject", "object-genuinely-wrong",
            "object-absent-unresolved"}


def interval_of(a):
    m = re.search(r"(\d{8}_\d{8})", a)
    return m.group(1) if m else "?"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]


rows = [r for r in load(RECLASS)
        if r.get("operational_class")
        and r["operational_class"] != "subject-unavailable"]
n = len(rows)

# ---- per-interval invalid rate ----
byint = defaultdict(list)
for r in rows:
    byint[interval_of(r["audit_id"])].append(r["operational_class"] in INVALID)

print("=== invalid rate by interval ===")
rates = {}
for k in sorted(byint):
    v = byint[k]
    w = wilson(sum(v), len(v))
    rates[k] = w.point
    print(f"  {k}  n={len(v):4d}  invalid {100*w.point:5.1f}% "
          f"[{100*w.lo:.1f},{100*w.hi:.1f}]")

# ---- leave-one-interval-out ----
print("\n=== leave-one-interval-out invalid rate ===")
loo = []
for drop in sorted(byint):
    kept = [x for k in byint if k != drop for x in byint[k]]
    rate = sum(kept) / len(kept)
    loo.append(rate)
    print(f"  drop {drop}: {100*rate:.2f}%")
print(f"  overall {100*sum(1 for r in rows if r['operational_class'] in INVALID)/n:.2f}%"
      f"   LOO range [{100*min(loo):.2f}, {100*max(loo):.2f}]")

# ---- unconditional decomposition over all 1500 ----
raw = load(RAW)
N = len(raw)
unres = [r for r in raw if r.get("operational_class") == "subject-unavailable"
         or not r.get("operational_class")]
reasons = Counter(r.get("error") or "unknown" for r in unres)
no_subj = reasons.get("no-subject-id", 0)

print(f"\n=== unconditional decomposition over all N={N} sampled rows ===")
def share(pred_count):
    w = wilson(pred_count, N)
    return f"{100*w.point:5.1f}% [{100*w.lo:.1f},{100*w.hi:.1f}]"

valid = sum(1 for r in rows if r["operational_class"] == "endpoint-valid")
nonuniq = sum(1 for r in rows if r["operational_class"] == "main-supported-nonunique")
misatt = sum(1 for r in rows if r["operational_class"] == "object-qid-misattached")
unusable = sum(1 for r in rows if r["operational_class"] in UNUSABLE)
print(f"  A. fully valid (unique main)     {share(valid)}")
print(f"  B. structurally valid, non-unique {share(nonuniq)}")
print(f"  C. valid text, wrong identifier  {share(misatt)}")
print(f"  D. unusable text supervision     {share(unusable)}")
print(f"  E. unresolvable release record   {share(len(unres))}  "
      f"(of which no-subject-id: {no_subj})")
print(f"\n  malformed unconditionally (C+D+E): "
      f"{share(misatt + unusable + len(unres))}")
print(f"  no-subject-id rows are themselves release defects; "
      f"D+E(no-subj) = {share(unusable + no_subj)}")
