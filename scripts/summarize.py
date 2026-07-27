"""Summarise an audit JSONL into prevalence estimates with Wilson intervals.

Usage:
    python scripts/summarize.py evidence/sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.stats import wilson  # noqa: E402

POPULATION = 502_382

# Which operational classes count as semantically invalid under the primary
# semantics. Kept explicit so the composition of the headline number is
# auditable rather than buried in a boolean.
INVALID = {
    "role-erased-qualifier",
    "role-erased-reference",
    "property-present-object-absent",
    "property-absent-on-subject",
}
ROLE_ERASURE = {"role-erased-qualifier", "role-erased-reference"}
IDENTITY = {"property-absent-on-subject"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--by", default="", help="extra stratifier: tag|property|interval")
    args = ap.parse_args()

    rows = []
    for line in open(args.path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    errs = [r for r in rows if r.get("error") and not r.get("operational_class")]
    ok = [r for r in rows if r.get("operational_class")]
    resolved = [r for r in ok if r["operational_class"] != "subject-unavailable"]

    print(f"records            : {len(rows)}")
    print(f"hard failures      : {len(errs)}")
    print(f"classified         : {len(ok)}")
    print(f"subject resolvable : {len(resolved)}")
    if not resolved:
        return

    n = len(resolved)
    cls = Counter(r["operational_class"] for r in resolved)

    print(f"\n=== operational class ({n} resolvable rows) ===")
    for k, v in cls.most_common():
        print(f"  {k:34s} {wilson(v, n).pct()}")

    print("\n=== headline prevalences ===")
    for name, keys in [
        ("semantically invalid (rho_WBE)", INVALID),
        ("  role erasure (qual/ref)", ROLE_ERASURE),
        ("  identity failure (prop absent)", IDENTITY),
        ("  value failure (prop present)", {"property-present-object-absent"}),
        ("endpoint valid (rho_WBE)", {"endpoint-valid"}),
        ("main-supported but non-unique", {"main-supported-nonunique"}),
        ("deprecated-only support", {"deprecated-only"}),
    ]:
        k = sum(cls[c] for c in keys)
        print(f"  {name:34s} {wilson(k, n).pct()}")

    # sensitivity semantics
    print("\n=== sensitivity semantics ===")
    for label, field in [("rho_all  (member of main set)", "valid_all_main"),
                         ("rho_truthy (best-rank member)", "valid_truthy")]:
        k = sum(1 for r in resolved if r.get(field))
        print(f"  {label:34s} {wilson(k, n).pct()}")

    print("\n=== identity-integrity signals ===")
    for label, fn in [
        ("property absent in every role", lambda r: not r.get("property_present_any_role")),
        ("property absent as main snak", lambda r: not r.get("property_present_main")),
        ("object id absent entirely", lambda r: not r.get("object_present_any_property")),
    ]:
        k = sum(1 for r in resolved if fn(r))
        print(f"  {label:34s} {wilson(k, n).pct()}")

    est = wilson(sum(cls[c] for c in INVALID), n)
    print(f"\nextrapolated to {POPULATION:,} released rows:")
    print(f"  invalid rows: {est.point*POPULATION:,.0f} "
          f"[{est.lo*POPULATION:,.0f}, {est.hi*POPULATION:,.0f}]")

    if args.by:
        key = {"tag": "tag", "interval": "source_interval",
               "property": "property_id"}[args.by]
        groups = defaultdict(list)
        for r in resolved:
            g = r.get(key)
            if isinstance(g, list):
                g = g[1][:10]
            groups[g].append(r)
        print(f"\n=== invalid rate by {args.by} ===")
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:25]:
            k = sum(1 for r in rs if r["operational_class"] in INVALID)
            print(f"  {str(g):26s} {wilson(k, len(rs)).pct()}")


if __name__ == "__main__":
    main()
