"""Permutation tests and fuller balance stats for the matched design.

Uses paired sign-flip (Rademacher) permutation tests, item-level and clustered by
relation, instead of Wilcoxon: the difference-in-differences terms are tied and
discrete, so a test that doesn't assume symmetry or magnitude ranking is more
appropriate. Also reports extra balance stats across the fuzzy/random strata
(baseline accuracy, subject-label length, target-subject overlap, answer length).
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "src")
from wbe_audit.llm import contains_answer, normalise  # noqa: E402

SEED = 20260723
B = 20000


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]


def subject_of(question: str) -> str:
    # "What is the <relation> of <subject>?"
    m = re.search(r" of (.+?)\?*$", question)
    return m.group(1) if m else question


def did_terms(rows, measure):
    out = []
    for r in rows:
        f_e = int(r["probes"]["fuzzy"][measure]["oracle"])
        g_e = int(r["probes"]["random"][measure]["oracle"])
        f_n = int(r["probes"]["fuzzy"][measure]["null"])
        g_n = int(r["probes"]["random"][measure]["null"])
        out.append(((f_e - g_e) - (f_n - g_n), r.get("edit_property_id", "?")))
    return out


def signflip_p(terms, by_relation, b=B, seed=SEED):
    rng = random.Random(seed)
    obs = statistics.mean(d for d, _ in terms)
    if by_relation:
        clusters = defaultdict(list)
        for d, rel in terms:
            clusters[rel].append(d)
        groups = list(clusters.values())
        n = sum(len(g) for g in groups)
        count = 0
        for _ in range(b):
            s = 0.0
            for g in groups:
                sign = 1 if rng.random() < 0.5 else -1
                s += sign * sum(g)
            if abs(s / n) >= abs(obs):
                count += 1
        return (count + 1) / (b + 1)
    else:
        ds = [d for d, _ in terms]
        n = len(ds)
        count = 0
        for _ in range(b):
            s = sum(d if rng.random() < 0.5 else -d for d in ds)
            if abs(s / n) >= abs(obs):
                count += 1
        return (count + 1) / (b + 1)


def analyse(path, label):
    rows = load(path)
    n = len(rows)
    print(f"\n########## {label}  (N={n}) ##########")

    # ---- extended balance ----
    print("=== extended balance across strata ===")
    for s in ("fuzzy", "random"):
        sims = [r["probes"][s]["similarity"] for r in rows]
        qlen = [len(r["probes"][s]["question"].split()) for r in rows]
        glen = [len(str(r["probes"][s]["gold"]).split()) for r in rows]
        slen = [len(subject_of(r["probes"][s]["question"]).split()) for r in rows]
        bacc = [contains_answer(r["probes"][s]["baseline"], r["probes"][s]["gold"])
                for r in rows]
        # target-token / probe-subject overlap
        ov = []
        for r in rows:
            tgt = set(normalise(r.get("counterfactual_answer", "")).split())
            subj = set(normalise(subject_of(r["probes"][s]["question"])).split())
            ov.append(1.0 if (tgt & subj) else 0.0)
        print(f"  {s:7s} sim {statistics.mean(sims):.3f}  base-acc "
              f"{100*statistics.mean(bacc):4.1f}%  subj-tok "
              f"{statistics.mean(slen):.2f}  q-tok {statistics.mean(qlen):.1f}  "
              f"ans-tok {statistics.mean(glen):.2f}  tgt-in-subj "
              f"{100*statistics.mean(ov):.1f}%")

    # ---- permutation tests ----
    for measure in ("leakage", "locality"):
        terms = did_terms(rows, measure)
        mean_d = statistics.mean(d for d, _ in terms)
        p_item = signflip_p(terms, by_relation=False)
        p_rel = signflip_p(terms, by_relation=True)
        print(f"=== {measure} DiD = {100*mean_d:+.2f}pp   "
              f"sign-flip p(item)={p_item:.2g}   p(rel-cluster)={p_rel:.2g}")


analyse("evidence/locality_qwen25_7b.jsonl", "Qwen2.5-7B reconstruction")
analyse("evidence/locality_llama31_8b.jsonl", "Llama3.1-8B reconstruction")
