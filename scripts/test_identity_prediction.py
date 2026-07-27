"""Falsifiable test of the identity-loss account.

If a released row names the *wrong* subject entity, the released property will
still often be present on that wrong entity whenever the property is common
(P31 instance of, P17 country, P131 located in). It will be absent when the
property is rare. So the identity-loss account predicts:

    property-present-object-absent  -> enriched in HIGH-frequency properties
    property-absent-on-subject      -> enriched in LOW-frequency properties

and both are then the same defect, split only by the base rate of the property.

The competing account -- these are ordinary stale or superseded values on the
*right* entity -- predicts no such relationship.

Property frequency is taken from the released benchmark itself (rows per
property), which is the relevant exposure.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.stats import wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("audit")
args = ap.parse_args()

# property exposure across the whole release
prop_rows = Counter()
for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
    for r in json.load(open(f, encoding="utf-8")):
        prop_rows[r.get("relation_id")] += 1
total = sum(prop_rows.values())

rows = [json.loads(l) for l in open(args.audit, encoding="utf-8") if l.strip()]
rows = [r for r in rows
        if r.get("operational_class")
        and r["operational_class"] != "subject-unavailable"]
print(f"{len(rows)} classified rows\n")

by_cls = defaultdict(list)
for r in rows:
    by_cls[r["operational_class"]].append(prop_rows.get(r["property_id"], 0))

print("=== property commonness by class (rows per property in the release) ===")
print(f"{'class':34s} {'n':>4s} {'median':>9s} {'mean':>9s}")
for cls, freqs in sorted(by_cls.items(), key=lambda kv: -len(kv[1])):
    if not freqs:
        continue
    print(f"{cls:34s} {len(freqs):4d} {statistics.median(freqs):9.0f} "
          f"{statistics.mean(freqs):9.0f}")

# same thing, binned
print("\n=== class composition by property-frequency tercile ===")
freqs_sorted = sorted(prop_rows.values())
def tercile(p: str) -> str:
    f = prop_rows.get(p, 0)
    if f >= 10000:
        return "common (>=10k rows)"
    if f >= 1000:
        return "mid (1k-10k)"
    return "rare (<1k)"

buckets = defaultdict(Counter)
for r in rows:
    buckets[tercile(r["property_id"])][r["operational_class"]] += 1

for b in ["common (>=10k rows)", "mid (1k-10k)", "rare (<1k)"]:
    c = buckets[b]
    n = sum(c.values())
    if not n:
        continue
    print(f"\n  {b}  (n={n})")
    for cls, k in c.most_common():
        print(f"    {cls:34s} {wilson(k, n).pct()}")

# the direct contrast
print("\n=== the prediction ===")
for b in ["common (>=10k rows)", "mid (1k-10k)", "rare (<1k)"]:
    c = buckets[b]
    n = sum(c.values())
    if not n:
        continue
    absent = c["property-absent-on-subject"]
    present = c["property-present-object-absent"]
    denom = absent + present
    share = f"{100*absent/denom:.1f}%" if denom else "n/a"
    print(f"  {b:22s} of the two identity-suspect classes, "
          f"property-ABSENT share = {share}  ({absent}/{denom})")
print("\n  identity-loss account predicts this share falls as properties get more common.")

# top properties in each suspect class
print("\n=== top properties per class ===")
for cls in ["property-absent-on-subject", "property-present-object-absent",
            "role-erased-qualifier", "endpoint-valid"]:
    c = Counter(r["property_id"] for r in rows if r["operational_class"] == cls)
    if not c:
        continue
    top = ", ".join(f"{p}({k})" for p, k in c.most_common(6))
    print(f"  {cls:34s} {top}")
