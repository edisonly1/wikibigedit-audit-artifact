"""Baseline descriptive stats over released WikiBigEdit interval files."""
import json
import glob
import os
from collections import Counter

rows_total = 0
tag_c = Counter()
prop_c = Counter()
dup_c = Counter()
probe_c = Counter()
per_file = []

for path in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
    d = json.load(open(path, encoding="utf-8"))
    rows_total += len(d)
    per_file.append((os.path.basename(path), len(d)))
    for i, r in enumerate(d):
        tag = r.get("tag")
        if isinstance(tag, float):  # released files contain literal NaN tags
            tag = "NaN"
        elif tag is None:
            tag = "null"
        elif tag == "":
            tag = "<empty>"
        tag_c[tag] += 1
        prop_c[r.get("relation_id")] += 1
        dup_c[(r.get("subject_id"), r.get("relation_id"), r.get("object_id"))] += 1
        probe_c["has_loc"] += bool(r.get("loc"))
        probe_c["has_mhop"] += bool(r.get("mhop"))
        probe_c["has_rephrase"] += bool(r.get("rephrase"))
        probe_c["has_personas"] += bool(r.get("personas"))

print(f"files: {len(per_file)}   rows: {rows_total}")
for f, n in per_file:
    print(f"  {f}  {n}")

print("\ntag distribution:")
for k, v in tag_c.most_common():
    print(f"  {k:12s} {v:8d}  ({100*v/rows_total:.1f}%)")

print("\nprobe availability:")
for k, v in probe_c.most_common():
    print(f"  {k:12s} {v:8d}  ({100*v/rows_total:.1f}%)")

print(f"\ndistinct (s,p,o) triples: {len(dup_c)}")
rep = sum(1 for v in dup_c.values() if v > 1)
print(f"triples appearing in >1 row: {rep}  "
      f"(rows involved: {sum(v for v in dup_c.values() if v > 1)})")

print(f"\ndistinct properties: {len(prop_c)}")
print("top 25 properties by row count:")
for k, v in prop_c.most_common(25):
    print(f"  {k:10s} {v:7d}  ({100*v/rows_total:.2f}%)")

# concentration
cum = 0
for i, (k, v) in enumerate(prop_c.most_common(), 1):
    cum += v
    if cum >= 0.5 * rows_total:
        print(f"\ntop {i} properties cover 50% of rows")
        break
