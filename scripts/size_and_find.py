"""Size the API workload and locate a likely role-erasure candidate.

One clear example: a town asked to have "held the position" of mayor, consistent
with a P39 (position held) qualifier on a P6 (head of government) statement being
flattened into a fact about the town.
"""
import glob
import json
import os
from collections import Counter

files = sorted(glob.glob("data/raw/wiki_big_edit_*.json"))

subj_interval = set()
rows = 0
p39 = []
per_prop_subjects = Counter()

for f in files:
    base = os.path.basename(f)
    d = json.load(open(f, encoding="utf-8"))
    for i, r in enumerate(d):
        rows += 1
        s = r.get("subject_id")
        subj_interval.add((base, s))
        if r.get("relation_id") == "P39":
            p39.append((base, i, r))

print(f"rows: {rows}")
print(f"distinct (interval, subject) pairs: {len(subj_interval)}")
print(f"  -> API requests for a full historical screen: {len(subj_interval)}")
print(f"  -> at 2.5 req/s that is {len(subj_interval)/2.5/3600:.1f} hours")
print(f"  -> for a 3600-row probability sample: ~3600 requests, "
      f"{3600/2.5/60:.0f} minutes")

print(f"\nP39 'position held' rows: {len(p39)}")
obj_c = Counter(r.get("object") for _, _, r in p39)
print("top P39 objects:")
for k, v in obj_c.most_common(15):
    print(f"  {str(k):40s} {v}")

mayor = [(f, i, r) for f, i, r in p39
         if isinstance(r.get("object"), str) and "mayor" in r["object"].lower()]
print(f"\nP39 rows whose object mentions 'mayor': {len(mayor)}")
for f, i, r in mayor[:10]:
    print(f"  {f}:{i}  {r['subject']} ({r['subject_id']}) "
          f"--{r['relation']}({r['relation_id']})--> {r['object']} ({r['object_id']})")
    print(f"      update Q: {r.get('update')}")
    print(f"      ans     : {r.get('ans')}")

# candidates whose subject looks like a place rather than a person: no reliable
# signal in the release itself, so just dump the distinct subjects for inspection
if mayor:
    subs = sorted({(r["subject"], r["subject_id"]) for _, _, r in mayor})
    print(f"\ndistinct 'mayor' subjects: {len(subs)}")
    for name, qid in subs[:40]:
        print(f"  {qid:12s} {name}")

out = "evidence/p39_mayor_candidates.json"
os.makedirs("evidence", exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump([{"file": f, "index": i, **r} for f, i, r in mayor], fh,
              ensure_ascii=False, indent=2)
print(f"\nwrote {out}")
