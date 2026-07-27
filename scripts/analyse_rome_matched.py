"""Analyse the ROME matched reconstruction: leakage/locality DiD with permutation
tests, the same estimator used for the in-context matched design."""
import json
import random
import statistics
import sys
from collections import defaultdict

from scipy import stats


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]


rows = load("evidence/rome_matched.jsonl")
succ = sum(r["edit_success"] for r in rows)
print(f"N={len(rows)}  edit success {succ}/{len(rows)} ({100*succ/len(rows):.1f}%)")

# recover each item's relation by replaying the seeded shuffle used at run time,
# so we can report relation-clustered inference alongside item-level bootstrap.
_items = json.load(open("evidence/probe_sets.json", encoding="utf-8"))
random.Random(20260723).shuffle(_items)
_items = _items[: len(rows)]
by_item = {i + 1: it for i, it in enumerate(_items)}
for r in rows:
    r["relation"] = by_item[r["item"]]["edit_property_id"]
rels = {r["relation"] for r in rows}
sizes = defaultdict(int)
for r in rows:
    sizes[r["relation"]] += 1
print(f"  {len(rels)} distinct relations; median {statistics.median(sizes.values()):.0f} "
      f"items/relation, max {max(sizes.values())}, "
      f"{sum(v == 1 for v in sizes.values())} singletons")


def did_terms(rows, measure):
    out = []
    for r in rows:
        f_e = int(r["probes"]["fuzzy"][measure]["edit"])
        g_e = int(r["probes"]["random"][measure]["edit"])
        f_n = int(r["probes"]["fuzzy"][measure]["null"])
        g_n = int(r["probes"]["random"][measure]["null"])
        out.append((f_e - g_e) - (f_n - g_n))
    return out


def signflip_p(d, b=20000, seed=20260723):
    rng = random.Random(seed)
    obs = statistics.mean(d)
    n = len(d)
    c = sum(1 for _ in range(b)
            if abs(sum(x if rng.random() < 0.5 else -x for x in d) / n) >= abs(obs))
    return (c + 1) / (b + 1)


def boot_ci(d, b=4000, seed=20260723):
    rng = random.Random(seed)
    n = len(d)
    est = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return est[int(.025 * b)], est[int(.975 * b)]


def cluster_boot_ci(d, clusters, b=4000, seed=20260723):
    # resample whole relations with replacement (block bootstrap)
    rng = random.Random(seed)
    groups = defaultdict(list)
    for x, c in zip(d, clusters):
        groups[c].append(x)
    keys = list(groups)
    est = []
    for _ in range(b):
        drawn = [groups[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [x for g in drawn for x in g]
        est.append(sum(flat) / len(flat))
    est.sort()
    return est[int(.025 * b)], est[int(.975 * b)]


def cluster_signflip_p(d, clusters, b=20000, seed=20260723):
    # flip the sign of every item in a relation together
    rng = random.Random(seed)
    groups = defaultdict(list)
    for x, c in zip(d, clusters):
        groups[c].append(x)
    keys = list(groups)
    obs = statistics.mean(d)
    n = len(d)
    c = 0
    for _ in range(b):
        s = 0.0
        for k in keys:
            sign = 1 if rng.random() < 0.5 else -1
            s += sign * sum(groups[k])
        if abs(s / n) >= abs(obs):
            c += 1
    return (c + 1) / (b + 1)


clusters = [r["relation"] for r in rows]
for measure in ("leakage", "locality"):
    d = did_terms(rows, measure)
    mean_d = statistics.mean(d)
    lo, hi = boot_ci(d)
    p = signflip_p(d)
    clo, chi = cluster_boot_ci(d, clusters)
    cp = cluster_signflip_p(d, clusters)
    print(f"  {measure:8s} DiD {100*mean_d:+.2f}pp | item boot[{100*lo:+.2f},"
          f"{100*hi:+.2f}] p={p:.2g} | relation-clustered boot[{100*clo:+.2f},"
          f"{100*chi:+.2f}] p={cp:.2g}")

for cond in ("edit", "null"):
    fr = sum(r["probes"]["fuzzy"]["leakage"][cond] for r in rows) / len(rows)
    rr = sum(r["probes"]["random"]["leakage"][cond] for r in rows) / len(rows)
    print(f"  raw leakage [{cond}]: fuzzy {100*fr:.1f}%  random {100*rr:.1f}%")
