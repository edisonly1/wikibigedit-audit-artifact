"""Formal trend test for the leakage dose-response (Figure 7).

Cochran-Armitage test for a monotone trend in leakage across ordered
probe-subject similarity bins, computed separately for the real-edit and the
unrelated (null) condition. Bin scores are the bin centres used in the figure.
"""
import json
import math

from scipy import stats

rows = [json.loads(l) for l in open("evidence/locality_qwen25_7b.jsonl",
                                    encoding="utf-8-sig") if l.strip()]
bins = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 1.01)]
centres = [0.15, 0.375, 0.525, 0.675, 0.88]


def counts(measure_key):
    x, n = [], []
    for lo, hi in bins:
        r = tot = 0
        for row in rows:
            for s in ("fuzzy", "random"):
                p = row["probes"][s]
                if lo <= p["similarity"] < hi:
                    tot += 1
                    r += bool(p["leakage"][measure_key])
        x.append(r)
        n.append(tot)
    return x, n


def cochran_armitage(x, n, t):
    N = sum(n)
    R = sum(x)
    p = R / N
    T = sum(t[i] * (x[i] - n[i] * p) for i in range(len(x)))
    snt = sum(n[i] * t[i] for i in range(len(x)))
    snt2 = sum(n[i] * t[i] * t[i] for i in range(len(x)))
    var = p * (1 - p) * (snt2 - snt * snt / N)
    z = T / math.sqrt(var)
    return z, 2 * stats.norm.sf(abs(z))


for key in ("oracle", "null"):
    x, n = counts(key)
    z, pv = cochran_armitage(x, n, centres)
    rates = [f"{100*xi/ni:.1f}" for xi, ni in zip(x, n)]
    print(f"{key:7s} bin leakage %: {rates}  (n per bin {n})")
    print(f"        Cochran-Armitage z={z:+.2f}  p={pv:.2g}")
