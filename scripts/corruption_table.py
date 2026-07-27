"""Exact counts, CIs, and the explicit contrast for the corruption analysis.

Among probes the model answered correctly BEFORE any edit, we compare gold-answer
retention under the row's own edit vs under an unrelated (null) edit. The paired
per-item difference (null retained minus edit retained) is the corruption; its
p-value is the paired sign test (exact binomial on discordant pairs).
"""
import json
import random
import sys

from scipy import stats

sys.path.insert(0, "src")
from wbe_audit.stats import wilson  # noqa: E402


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]


def paired_loss_ci(rows, probe, b=8000, seed=20260723):
    # paired bootstrap over items of loss = retain(null) - retain(edit)
    rng = random.Random(seed)
    pairs = [(int(r["probes"][probe]["gold_kept"]["null"]),
              int(r["probes"][probe]["gold_kept"]["edit"])) for r in rows]
    n = len(pairs)
    est = []
    for _ in range(b):
        s = 0
        for _ in range(n):
            nu, ed = pairs[rng.randrange(n)]
            s += nu - ed
        est.append(100 * s / n)
    est.sort()
    return est[int(.025 * b)], est[int(.975 * b)]


def paired_p(rows, probe):
    # discordant pairs: retained under null but lost under edit (b) vs reverse (c)
    b = sum(1 for r in rows if r["probes"][probe]["gold_kept"]["null"]
            and not r["probes"][probe]["gold_kept"]["edit"])
    c = sum(1 for r in rows if not r["probes"][probe]["gold_kept"]["null"]
            and r["probes"][probe]["gold_kept"]["edit"])
    n = b + c
    p = stats.binomtest(b, n, 0.5).pvalue if n else 1.0
    return b, c, p


for path, label in [("evidence/released_exp_qwen2.5_7b.jsonl", "Qwen-7B"),
                    ("evidence/released_exp_llama3.1_8b.jsonl", "Llama-8B")]:
    rows = load(path)
    print(f"\n### {label} ###")
    for probe in ("own", "other"):
        bc = [r for r in rows if r["probes"][probe]["baseline_correct"]]
        n = len(bc)
        ke = sum(r["probes"][probe]["gold_kept"]["edit"] for r in bc)
        kn = sum(r["probes"][probe]["gold_kept"]["null"] for r in bc)
        we, wn = wilson(ke, n), wilson(kn, n)
        b, c, p = paired_p(bc, probe)
        drop = 100 * (kn - ke) / n
        dlo, dhi = paired_loss_ci(bc, probe)
        print(f"  {probe:5s} n={n:3d}  retain(null) {kn}/{n}="
              f"{100*wn.point:4.1f}% [{100*wn.lo:.1f},{100*wn.hi:.1f}]  "
              f"retain(edit) {ke}/{n}={100*we.point:4.1f}% "
              f"[{100*we.lo:.1f},{100*we.hi:.1f}]  loss {drop:+.1f}pp "
              f"[{dlo:+.1f},{dhi:+.1f}]  discordant {b}/{c}  p={p:.2g}")
