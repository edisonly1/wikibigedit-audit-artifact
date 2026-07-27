"""What explains the 'property present, object absent' rows?

This is the largest non-valid block (~28% at t1) and it is the one the paper has
to account for. WikiBigEdit extracted the pairing from the t1 snapshot, so the
object ought to be there. Two mechanisms compete:

  A. window misassociation -- the scanner read the property from one snak and the
     object id from a neighbouring one, so the object IS present in the entity but
     under a different property;

  B. wrong entity -- the released subject_id is not the entity the pairing came
     from, so the object is absent from it entirely.

`object_present_any_property` separates them, and it is already recorded on every
audited row. No API access needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.stats import wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
args = ap.parse_args()

rows = []
for p in args.paths:
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("operational_class") and \
                    r["operational_class"] != "subject-unavailable":
                rows.append(r)
print(f"{len(rows)} classified rows from {len(args.paths)} file(s)\n")

print("=== is the released object present anywhere in the subject entity? ===")
print(f"{'operational class':34s} {'n':>5s}  object present under SOME property")
for cls in sorted({r["operational_class"] for r in rows}):
    sub = [r for r in rows if r["operational_class"] == cls]
    k = sum(1 for r in sub if r.get("object_present_any_property"))
    print(f"  {cls:32s} {len(sub):5d}  {wilson(k, len(sub)).pct()}")

target = [r for r in rows
          if r["operational_class"] == "property-present-object-absent"]
if target:
    k = sum(1 for r in target if r.get("object_present_any_property"))
    n = len(target)
    print(f"\n=== the {n} 'property present, object absent' rows ===")
    print(f"  A. object present under a DIFFERENT property "
          f"(window misassociation): {wilson(k, n).pct()}")
    print(f"  B. object absent from the entity entirely "
          f"(wrong entity):          {wilson(n - k, n).pct()}")

# same question for the identity-failure class, as a control: if those rows name
# the wrong entity, the object should almost never be present either
ctrl = [r for r in rows if r["operational_class"] == "property-absent-on-subject"]
if ctrl:
    k = sum(1 for r in ctrl if r.get("object_present_any_property"))
    print(f"\n=== control: {len(ctrl)} 'property absent on subject' rows ===")
    print(f"  object present under some property: {wilson(k, len(ctrl)).pct()}")
    print("  (should be low if these name the wrong entity)")

valid = [r for r in rows if r["operational_class"] == "endpoint-valid"]
if valid:
    k = sum(1 for r in valid if r.get("object_present_any_property"))
    print(f"\n=== control: {len(valid)} endpoint-valid rows ===")
    print(f"  object present under some property: {wilson(k, len(valid)).pct()}")
    print("  (should be ~100% by definition -- sanity check on the field)")

# how many main statements does the subject carry for the released property?
print("\n=== number of main statements for the released property ===")
for cls in ["property-present-object-absent", "endpoint-valid",
            "main-supported-nonunique"]:
    sub = [r for r in rows if r["operational_class"] == cls]
    if not sub:
        continue
    c = Counter(r.get("n_main_statements", 0) for r in sub)
    tot = len(sub)
    dist = ", ".join(f"{k}:{100*v/tot:.0f}%" for k, v in sorted(c.items())[:6])
    print(f"  {cls:34s} {dist}")
