"""Endpoint-semantics sensitivity and the supervision/metadata split.

Reports rho_WBE (singleton), rho_all (membership) and rho_truthy (best-rank) side by
side, and separates unusable text, valid-text/wrong-id, and
structurally-valid-but-non-unique. Intervals are the relation-clustered bootstrap
(the conservative choice, given the design effect from cluster_bootstrap.py).
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, "src")
from wbe_audit.stats import wilson  # noqa: E402

RECLASS = "evidence/hist_n1500.reclassified.jsonl"
SEED = 20260722
B = 4000


def interval_of(a):
    m = re.search(r"(\d{8}_\d{8})", a)
    return m.group(1) if m else "?"


def rel_boot(rows, pred, b=B):
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["property_id"]].append(bool(pred(r)))
    keys = list(clusters)
    rng = random.Random(SEED)
    ests = []
    for _ in range(b):
        num = den = 0
        for _ in range(len(keys)):
            c = clusters[keys[rng.randrange(len(keys))]]
            num += sum(c); den += len(c)
        ests.append(num / den if den else 0)
    ests.sort()
    return ests[int(0.025 * b)], ests[int(0.975 * b)]


rows = [r for r in (json.loads(l) for l in open(RECLASS, encoding="utf-8-sig")
                    if l.strip())
        if r.get("operational_class")
        and r["operational_class"] != "subject-unavailable"]
n = len(rows)


def report(name, pred):
    k = sum(1 for r in rows if pred(r))
    w = wilson(k, n)
    lo, hi = rel_boot(rows, pred)
    print(f"  {name:38s} {100*w.point:5.1f}  Wilson[{100*w.lo:4.1f},{100*w.hi:4.1f}]"
          f"  rel-clu[{100*lo:4.1f},{100*hi:4.1f}]")


print(f"n = {n} resolvable rows\n")
print("=== endpoint-semantics sensitivity (share VALID under each definition) ===")
report("rho_WBE  (unique non-dep main = {o})",
       lambda r: r["operational_class"] == "endpoint-valid")
report("rho_all  (o in non-dep main set)", lambda r: r.get("valid_all_main"))
report("rho_truthy (o in best-rank main set)", lambda r: r.get("valid_truthy"))

print("\n=== three-way decomposition of the released rows ===")
report("A. fully valid (unique main)",
       lambda r: r["operational_class"] == "endpoint-valid")
report("B. structurally valid, non-unique",
       lambda r: r["operational_class"] == "main-supported-nonunique")
report("C. valid text, wrong structured id",
       lambda r: r["operational_class"] == "object-qid-misattached")
UNUSABLE = {"role-erased-qualifier", "role-erased-reference",
            "property-absent-on-subject", "object-genuinely-wrong",
            "object-absent-unresolved"}
report("D. unusable text supervision",
       lambda r: r["operational_class"] in UNUSABLE)

print("\n  A+B = structurally valid (rho_all):",
      f"{100*sum(1 for r in rows if r['operational_class'] in {'endpoint-valid','main-supported-nonunique'})/n:.1f}%")
print("  A+B+C = text supervision defensible:",
      f"{100*sum(1 for r in rows if r['operational_class'] in {'endpoint-valid','main-supported-nonunique','object-qid-misattached'})/n:.1f}%")
