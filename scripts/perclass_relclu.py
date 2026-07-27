import json
import random
import sys
from collections import defaultdict

sys.path.insert(0, "src")
from wbe_audit.stats import wilson  # noqa: E402

rows = [r for r in (json.loads(l) for l in
        open("evidence/hist_n1500.reclassified.jsonl", encoding="utf-8-sig")
        if l.strip())
        if r.get("operational_class")
        and r["operational_class"] != "subject-unavailable"]
n = len(rows)


def relboot(pred, b=4000, seed=20260722):
    cl = defaultdict(list)
    for r in rows:
        cl[r["property_id"]].append(bool(pred(r)))
    keys = list(cl)
    rng = random.Random(seed)
    est = []
    for _ in range(b):
        num = den = 0
        for _ in range(len(keys)):
            c = cl[keys[rng.randrange(len(keys))]]
            num += sum(c)
            den += len(c)
        est.append(num / den if den else 0)
    est.sort()
    return est[int(.025 * b)], est[int(.975 * b)]


for c in ["endpoint-valid", "object-qid-misattached", "property-absent-on-subject",
          "role-erased-qualifier", "main-supported-nonunique",
          "object-absent-unresolved", "object-genuinely-wrong",
          "role-erased-reference"]:
    def pred(r, c=c):
        return r["operational_class"] == c
    k = sum(1 for r in rows if pred(r))
    w = wilson(k, n)
    lo, hi = relboot(pred)
    print(f"{c:32s} {100*w.point:5.1f}  W[{100*w.lo:4.1f},{100*w.hi:4.1f}]  "
          f"rel[{100*lo:4.1f},{100*hi:4.1f}]")
