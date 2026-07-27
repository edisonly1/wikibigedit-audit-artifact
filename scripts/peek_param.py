import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "evidence/smoke_param.jsonl"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rows = [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]
print(f"{len(rows)} records; edit success "
      f"{sum(r['edit_success'] for r in rows)}/{len(rows)}\n")
for r in rows[:k]:
    print("=" * 70)
    print(f"EDIT: {r['subject']} | {r['relation']} -> {r['ans']}  "
          f"success={r['edit_success']}")
    for name, p in r["probes"].items():
        print(f"  [{name}] gold={p['gold']!r} base_correct={p['baseline_correct']}")
        print(f"     base : {p['baseline']!r}")
        for c in ("edit", "null"):
            print(f"     {c:5s}: {p['answers'][c]!r:34s} "
                  f"loc={p['locality'][c]} leak={p['leakage'][c]} "
                  f"goldkept={p['gold_kept'][c]}")
