"""PPV and false-omission rate for the classifier, with Clopper-Pearson intervals.

Taking "invalid under rho_WBE" as the positive class, PPV is P(independent parser
says invalid | classifier says invalid) and FOR is the same given the classifier
says valid. The comparator is the independent parser over the same pinned revision,
so this measures whether the classifier's verdict is faithful to the actual Wikidata
state, which (the verdict being a deterministic function of that state) is its
precision.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")
from wbe_audit.stats import clopper_pearson, wilson  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "evidence/extraction_verification.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]
ok = [r for r in rows if r.get("status") != "fetch-failed"
      and "independent_verdict" in r]
print(f"{len(rows)} rows, {len(ok)} verified ({len(rows)-len(ok)} fetch-failed)\n")

called_invalid = [r for r in ok if r["stored_verdict"] == "invalid"]
called_valid = [r for r in ok if r["stored_verdict"] == "valid"]

ppv_k = sum(1 for r in called_invalid if r["independent_verdict"] == "invalid")
for_k = sum(1 for r in called_valid if r["independent_verdict"] == "invalid")

print("=== classifier verdict vs independent re-parse of the pinned revision ===")
print(f"  classifier 'invalid' n = {len(called_invalid)}")
if called_invalid:
    cp = clopper_pearson(ppv_k, len(called_invalid))
    print(f"    PPV (truly invalid)          : {100*cp.point:.1f}% "
          f"[{100*cp.lo:.1f}, {100*cp.hi:.1f}]  ({ppv_k}/{len(called_invalid)})")
print(f"  classifier 'valid'   n = {len(called_valid)}")
if called_valid:
    cp = clopper_pearson(for_k, len(called_valid))
    print(f"    false-omission rate          : {100*cp.point:.1f}% "
          f"[{100*cp.lo:.1f}, {100*cp.hi:.1f}]  ({for_k}/{len(called_valid)})")

# overall agreement
agree = sum(1 for r in ok if r["verdict_agree"])
print(f"\n  overall verdict agreement      : {wilson(agree, len(ok)).pct()}")

# role-erasure sub-check
role = [r for r in ok if r.get("role_agree") is not None]
if role:
    rk = sum(1 for r in role if r["role_agree"])
    print(f"  role-erasure agreement         : {wilson(rk, len(role)).pct()}")

dis = [r for r in ok if not r["verdict_agree"]]
if dis:
    print(f"\n  disagreements ({len(dis)}):")
    for r in dis:
        print(f"    {r['audit_id']}: stored={r['stored_verdict']} "
              f"independent={r['independent_verdict']} class={r['stored_class']}")
