"""Build the repaired, provenance-bearing WikiBigEdit resource from audited rows.

Emits three tracks plus per-row evidence, so the resource is usable both as a
cleaner benchmark and as an error-analysis corpus:

  atomic       endpoint-valid rows: unique, role-correct main facts
  endpoint     main-supported-nonunique rows: role-correct but not unique
               (valid under rho_all; retained with a multiplicity flag)
  invalid      role-erased / identity-absent / object-metadata rows, retained WITH
               the diagnosis and, where available, the corrected object QID

Every row carries its audit_id, snapshot revision, endpoint answer sets, role
evidence, and operational class, so any downstream user can re-derive the verdict.

Only rows that were actually audited (appear in the reclassified jsonl) are
emitted; the file records how many that is against the full release so coverage is
never overstated.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

TRACK = {
    "endpoint-valid": "atomic",
    "main-supported-nonunique": "endpoint",
    "role-erased-qualifier": "invalid",
    "role-erased-reference": "invalid",
    "property-absent-on-subject": "invalid",
    "object-qid-misattached": "invalid",
    "object-genuinely-wrong": "invalid",
    "object-absent-unresolved": "invalid",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit", help="reclassified audit jsonl")
    ap.add_argument("--out", default="release/wikibigedit_repaired.jsonl")
    ap.add_argument("--meta", default="release/repaired_manifest.json")
    args = ap.parse_args()

    release = {}
    total_release = 0
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        release[os.path.basename(f)] = d
        total_release += len(d)

    rows = [json.loads(l) for l in open(args.audit, encoding="utf-8-sig")
            if l.strip()]
    audited = [r for r in rows if r.get("operational_class")
               and r["operational_class"] != "subject-unavailable"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    track_counts = Counter()
    corrected = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in audited:
            cls = r["operational_class"]
            track = TRACK.get(cls, "invalid")
            track_counts[track] += 1

            base = r["audit_id"].split(":")[0]
            idx = r.get("source_row_index")
            src = (release.get(base, [None])[idx]
                   if idx is not None and base in release
                   and idx < len(release[base]) else {})

            corrected_object = None
            if cls == "object-qid-misattached" and r.get("actual_object_best_qid"):
                corrected_object = {
                    "object_id": r["actual_object_best_qid"],
                    "object_label": r.get("actual_object_best_label"),
                    "label_similarity": r.get("object_label_similarity"),
                }
                corrected += 1

            out = {
                "audit_id": r["audit_id"],
                "track": track,
                "operational_class": cls,
                "endpoint_verdict_rho_WBE": r.get("endpoint_primary"),
                "valid_rho_all": r.get("valid_all_main"),
                "valid_rho_truthy": r.get("valid_truthy"),
                "released": {
                    "subject": src.get("subject") if src else None,
                    "subject_id": r.get("subject_id"),
                    "relation": src.get("relation") if src else None,
                    "relation_id": r.get("property_id"),
                    "object": src.get("object") if src else None,
                    "object_id": r.get("target_object_id"),
                    "tag": src.get("tag") if src else None,
                },
                "corrected_object": corrected_object,
                "evidence": {
                    "snapshot_revision": r.get("revision_id"),
                    "snapshot_timestamp": r.get("revision_timestamp"),
                    "rho_WBE_main_answer_set": r.get("endpoint_primary_objects"),
                    "support_roles": r.get("support_roles"),
                    "qualifier_of": r.get("qualifier_of"),
                    "reference_of": r.get("reference_of"),
                    "property_present_main": r.get("property_present_main"),
                    "object_present_any_property": r.get("object_present_any_property"),
                },
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    manifest = {
        "source": "lukasthede/WikiBigEdit, 8 interval files",
        "total_released_rows": total_release,
        "audited_rows": len(audited),
        "coverage_fraction": round(len(audited) / total_release, 5),
        "tracks": dict(track_counts),
        "object_qids_corrected": corrected,
        "classifier": "frozen at Phase-A; see notes/11_phaseA_and_freeze.md",
        "caveat": ("Tracks are computed on the audited probability sample, not the "
                   "full 502,382 rows. Prevalences generalise; per-row labels exist "
                   "only for audited rows. A full deterministic screen is future "
                   "work."),
    }
    os.makedirs(os.path.dirname(args.meta) or ".", exist_ok=True)
    json.dump(manifest, open(args.meta, "w", encoding="utf-8"), indent=1)

    print(f"wrote {args.out}  ({len(audited)} rows)")
    print(f"wrote {args.meta}")
    print(f"\ncoverage: {len(audited)}/{total_release} "
          f"({100*len(audited)/total_release:.3f}% of the release)")
    for t, c in track_counts.most_common():
        print(f"  {t:10s} {c:5d}  ({100*c/len(audited):.1f}%)")
    print(f"  object QIDs corrected: {corrected}")


if __name__ == "__main__":
    main()
