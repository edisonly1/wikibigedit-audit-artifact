"""Build the sampled-rows manifest for the artifact release.

One record per sampled row, covering the reviewer's checklist items 1-3 and 5:
sampled row identity, inclusion status, audit label, pinned revision id, and the
corrected object identifier where one was assigned. Everything here is derived
deterministically from the released benchmark plus the pinned Wikidata revisions,
so a third party can reproduce it from the seed alone.
"""
from __future__ import annotations

import json
import re

RAW = "evidence/hist_n1500.jsonl"                 # all 1500 sampled, incl. unresolved
RECLASS = "evidence/hist_n1500.reclassified.jsonl"  # 1458 resolved, final classes
OUT = "release/sampled_rows_manifest.jsonl"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]


def interval_of(a):
    m = re.search(r"(\d{8}_\d{8})", a or "")
    return m.group(1) if m else None


reclass = {r["audit_id"]: r for r in load(RECLASS) if r.get("audit_id")}

n = 0
with open(OUT, "w", encoding="utf-8") as fh:
    for r in load(RAW):
        aid = r.get("audit_id")
        resolvable = bool(r.get("operational_class")
                          and r["operational_class"] != "subject-unavailable")
        rc = reclass.get(aid, {})
        rec = {
            "audit_id": aid,
            "subject_id": r.get("subject_id"),
            "property_id": r.get("property_id"),
            "object_id": r.get("target_object_id"),
            "source_interval": interval_of(aid),
            "source_row_index": r.get("source_row_index"),
            "resolvable": resolvable,
            "unresolvable_reason": (None if resolvable else r.get("error")),
            "pinned_revision_id": r.get("revision_id"),
            "revision_timestamp": r.get("revision_timestamp"),
            "operational_class": (rc.get("operational_class")
                                  if resolvable else None),
            "endpoint_verdict_rho_WBE": (rc.get("endpoint_primary")
                                         if resolvable else None),
            "valid_rho_all": rc.get("valid_all_main") if resolvable else None,
            "valid_rho_truthy": rc.get("valid_truthy") if resolvable else None,
            # a "correction" applies only to misattached-identifier rows, matching
            # the 337 reported in the paper; for other classes we still record the
            # entity's actual main value under resolved_actual_object for analysis.
            "corrected_object_id": (
                rc.get("actual_object_best_qid")
                if rc.get("operational_class") == "object-qid-misattached"
                else None),
            "resolved_actual_object_id": rc.get("actual_object_best_qid"),
            "resolved_actual_object_label": rc.get("actual_object_best_label"),
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1

print(f"wrote {OUT}  ({n} rows)")
resolvable = sum(1 for l in open(OUT, encoding="utf-8")
                 if json.loads(l)["resolvable"])
corrected = sum(1 for l in open(OUT, encoding="utf-8")
                if json.loads(l)["corrected_object_id"])
print(f"  resolvable: {resolvable}   unresolvable: {n - resolvable}   "
      f"corrected identifiers: {corrected}")
