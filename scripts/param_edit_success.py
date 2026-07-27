import json
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "evidence/param_exp_ftl.jsonl"
rows = [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
k = sum(1 for r in rows if r.get("edit_success"))
print(f"edit success: {k}/{len(rows)} ({100*k/len(rows):.1f}%)")
