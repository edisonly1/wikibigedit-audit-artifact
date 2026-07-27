"""Inspect the structure of a released WikiBigEdit interval file."""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/wiki_big_edit_20240201_20240220.json"
d = json.load(open(path, encoding="utf-8"))

print("top-level type:", type(d).__name__)
if isinstance(d, dict):
    print("top-level keys:", list(d.keys()))
    for k, v in d.items():
        n = len(v) if hasattr(v, "__len__") else "NA"
        print(f"  {k}: {type(v).__name__}  len={n}")
        if isinstance(v, list) and v:
            print("    first:", json.dumps(v[0], ensure_ascii=False)[:900])
        elif isinstance(v, dict) and v:
            fk = next(iter(v))
            print(f"    first key {fk!r}:", json.dumps(v[fk], ensure_ascii=False)[:900])
else:
    print("len:", len(d))
    print("first:", json.dumps(d[0], ensure_ascii=False)[:900])
    if isinstance(d[0], dict):
        print("row keys:", list(d[0].keys()))
        print("\nkey presence over first 5000 rows:")
        c = Counter()
        for r in d[:5000]:
            c.update(r.keys())
        for k, n in c.most_common():
            print(f"  {k:30s} {n}")
