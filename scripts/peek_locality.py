"""Print a few raw experiment records for eyeball inspection."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "evidence/smoke_loc.jsonl"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 2

rows = [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]
print(f"{len(rows)} records\n")
for r in rows[:k]:
    print("=" * 78)
    print("EDIT Q :", r["edit_question"])
    print("true   :", r["edit_true_answer"], "| counterfactual:",
          r["counterfactual_answer"])
    print("baseline answer to edit Q:", repr(r["edit_success"]["baseline_answer"]))
    print("edit success (oracle)    :", r["edit_success"]["oracle"])
    for s, p in r["probes"].items():
        print(f"\n  [{s}]  sim={p['similarity']}  own_retrieved={p['retrieved_is_own']}"
              f"  retr_sim={p['retrieval_sim']}  gate={p['scope_gate_yes']}")
        print(f"    Q    : {p['question']}")
        print(f"    gold : {p['gold']}")
        print(f"    base : {p['baseline']!r}")
        for m in ("oracle", "retrieval", "scope_gated", "null"):
            print(f"    {m:11s} ans={p['answers'][m]!r:38s} "
                  f"loc={p['locality'][m]}  leak={p['leakage'][m]}")
