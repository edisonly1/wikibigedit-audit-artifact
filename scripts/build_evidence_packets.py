"""Build human-adjudicable evidence packets from audited rows.

Each packet has everything an adjudicator needs to decide a row's true class without
re-querying: the released row, the endpoint answer sets, the role decomposition, the
snapshot revision, and the model's proposed class with its evidence. Packets are
stratified across operational classes so every class is checked. The verdict goes in
`human_class` (blank here) and score_adjudication.py compares the two.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import glob  # noqa: E402


def load_release_index():
    idx = {}
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        idx[os.path.basename(f)] = json.load(open(f, encoding="utf-8"))
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit", help="reclassified audit jsonl")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", default="evidence/phaseA_packets.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.audit, encoding="utf-8-sig")
            if l.strip()]
    rows = [r for r in rows if r.get("operational_class")
            and r["operational_class"] != "subject-unavailable"]
    release = load_release_index()

    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["operational_class"]].append(r)

    rng = random.Random(args.seed)
    packets = []
    for cls, rs in sorted(by_cls.items()):
        for r in rng.sample(rs, min(args.per_class, len(rs))):
            base = r["audit_id"].split(":")[0]
            idx = r.get("source_row_index")
            released = (release.get(base, [None])[idx]
                       if idx is not None and base in release
                       and idx < len(release[base]) else None)
            packets.append({
                "audit_id": r["audit_id"],
                "released_row": {
                    "subject": released.get("subject") if released else None,
                    "subject_id": r.get("subject_id"),
                    "relation": released.get("relation") if released else None,
                    "relation_id": r.get("property_id"),
                    "object": released.get("object") if released else None,
                    "object_id": r.get("target_object_id"),
                    "tag": released.get("tag") if released else None,
                    "update_question": released.get("update") if released else None,
                    "ans": released.get("ans") if released else None,
                },
                "evidence": {
                    "snapshot_revision": r.get("revision_id"),
                    "snapshot_timestamp": r.get("revision_timestamp"),
                    "rho_WBE_main_answer_set": r.get("endpoint_primary_objects"),
                    "rho_truthy_answer_set": r.get("endpoint_truthy_objects"),
                    "n_main_statements": r.get("n_main_statements"),
                    "support_roles": r.get("support_roles"),
                    "qualifier_of_statements": r.get("qualifier_of"),
                    "reference_of_statements": r.get("reference_of"),
                    "property_present_any_role": r.get("property_present_any_role"),
                    "property_present_main": r.get("property_present_main"),
                    "object_present_any_property": r.get("object_present_any_property"),
                    "actual_object_best_qid": r.get("actual_object_best_qid"),
                    "actual_object_best_label": r.get("actual_object_best_label"),
                    "object_label_similarity": r.get("object_label_similarity"),
                },
                "model_class": r["operational_class"],
                "model_endpoint_verdict": r.get("endpoint_primary"),
                "human_class": "",          # <-- adjudicator fills this
                "human_notes": "",
            })

    rng.shuffle(packets)          # blind the adjudicator to class order
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(packets, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"wrote {len(packets)} packets to {args.out}")
    print("classes represented:",
          dict((c, len(v)) for c, v in sorted(by_cls.items())))


if __name__ == "__main__":
    main()
