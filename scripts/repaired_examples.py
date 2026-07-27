"""Pull a few clean repaired-record examples for the paper appendix."""
import json

rows = [json.loads(l) for l in open("release/wikibigedit_repaired.jsonl",
                                     encoding="utf-8-sig") if l.strip()]


def show(r):
    rel = r["released"]
    print(f"class={r['operational_class']}  rev={r['evidence']['snapshot_revision']}")
    print(f"  released : {rel['subject']} ({rel['subject_id']}) "
          f"--{rel['relation']}({rel['relation_id']})--> "
          f"{rel['object']} ({rel['object_id']})")
    if r.get("corrected_object"):
        co = r["corrected_object"]
        print(f"  corrected: object_id {rel['object_id']} -> {co['object_id']} "
              f"(label {co['object_label']}, sim {co['label_similarity']})")
    print(f"  roles={r['evidence']['support_roles']} "
          f"qual_of={r['evidence']['qualifier_of']} "
          f"ref_of={r['evidence']['reference_of']}  "
          f"rho_WBE_set={r['evidence']['rho_WBE_main_answer_set']}")
    print()


print("=== object-qid-misattached (corrected) ===")
shown = 0
for r in rows:
    if r["operational_class"] == "object-qid-misattached" and r.get("corrected_object"):
        show(r); shown += 1
        if shown >= 4:
            break

print("=== role-erased-qualifier ===")
shown = 0
for r in rows:
    if r["operational_class"] == "role-erased-qualifier":
        show(r); shown += 1
        if shown >= 3:
            break
