"""Conservative measurement of released `subject` corruption.

The first pass (subject_corruption.py) over-counted, mostly from a length rule that
fires on long titles. Here we keep only signatures that can't be a real entity name:
S1, the subject is a single function word (the rewrite collapsed to the question's
first word); S2, one QID carries substantially different subject strings (difflib
ratio < 0.90, so dash/punctuation variants don't count); and S3, the subject doesn't
occur in its own question. Computed from the release alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.stats import wilson  # noqa: E402

FUNCTION_WORDS = {"what", "which", "who", "whom", "whose", "where", "when",
                  "why", "how", "in", "of", "the", "is", "are", "was", "were",
                  "did", "does", "do", "has", "have", "had", "a", "an", "by",
                  "for", "at", "on", "to", "with", "and", "or", "as", "from"}


def norm(s: str) -> str:
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", s.lower())).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/subject_corruption_strict.json")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        base = os.path.basename(f)
        for i, r in enumerate(json.load(open(f, encoding="utf-8"))):
            rows.append((base, i, r))
    n = len(rows)
    print(f"{n} released rows\n")

    by_qid = defaultdict(set)
    for _, _, r in rows:
        s, sl = r.get("subject_id"), r.get("subject")
        if s and isinstance(sl, str):
            by_qid[s].add(sl)

    # S2: QIDs whose subject strings genuinely diverge
    divergent = {}
    for qid, labels in by_qid.items():
        if len(labels) < 2:
            continue
        ls = sorted(labels)
        worst = 1.0
        pair = None
        for a in range(len(ls)):
            for b in range(a + 1, len(ls)):
                rr = SequenceMatcher(None, norm(ls[a]), norm(ls[b])).ratio()
                if rr < worst:
                    worst, pair = rr, (ls[a], ls[b])
        if worst < 0.90:
            divergent[qid] = {"labels": ls, "min_ratio": worst, "pair": pair}

    c = Counter()
    examples = defaultdict(list)
    per_interval = defaultdict(Counter)

    for base, i, r in rows:
        subj = r.get("subject")
        q = r.get("update")
        qid = r.get("subject_id")
        if not isinstance(subj, str):
            c["S0-subject-not-string"] += 1
            continue
        hit = False
        toks = subj.split()
        if len(toks) == 1 and subj.strip().lower() in FUNCTION_WORDS:
            c["S1-collapsed-to-function-word"] += 1
            per_interval[base]["S1"] += 1
            hit = True
            if len(examples["S1"]) < 200:
                examples["S1"].append({"subject": subj, "qid": qid, "q": q,
                                       "ans": r.get("ans")})
        if qid in divergent:
            c["S2-divergent-labels-for-one-qid"] += 1
            per_interval[base]["S2"] += 1
            hit = True
            if len(examples["S2"]) < 200:
                examples["S2"].append({"subject": subj, "qid": qid, "q": q,
                                       "all_labels": divergent[qid]["labels"][:5],
                                       "min_ratio": round(divergent[qid]["min_ratio"], 3)})
        if isinstance(q, str) and subj and subj not in q:
            c["S3-absent-from-own-question"] += 1
            per_interval[base]["S3"] += 1
            hit = True
            if len(examples["S3"]) < 200:
                examples["S3"].append({"subject": subj, "qid": qid, "q": q})
        if hit:
            c["ANY"] += 1

    print("=== conservative corruption signatures ===")
    for k in ["S1-collapsed-to-function-word", "S2-divergent-labels-for-one-qid",
              "S3-absent-from-own-question", "ANY"]:
        print(f"  {k:34s} {wilson(c[k], n).pct()}")

    print(f"\n  QIDs with divergent subject strings: {len(divergent)} of "
          f"{len(by_qid)} ({100*len(divergent)/len(by_qid):.2f}%)")

    print("\n=== by interval (ANY signature) ===")
    for base in sorted(per_interval):
        tot = sum(1 for b, _, _ in rows if b == base)
        k = sum(per_interval[base].values())
        print(f"  {base[14:-5]:20s} <= {100*k/tot:5.2f}%  (flag count {k} / {tot})")

    print("\n=== S2 examples: one QID, divergent subject strings ===")
    seen = set()
    shown = 0
    for e in examples["S2"]:
        if e["qid"] in seen or shown >= 6:
            continue
        seen.add(e["qid"])
        shown += 1
        print(f"\n  {e['qid']}  (min pairwise ratio {e['min_ratio']})")
        for lab in e["all_labels"]:
            print(f"    - {lab!r}")

    print("\n=== S3 examples: subject absent from its own question ===")
    for e in examples["S3"][:5]:
        print(f"  subject={e['subject']!r}")
        print(f"    Q: {e['q']}")

    json.dump({"counts": dict(c), "n_rows": n,
               "n_divergent_qids": len(divergent), "n_qids": len(by_qid),
               "divergent_sample": dict(list(divergent.items())[:200]),
               "examples": {k: v[:80] for k, v in examples.items()}},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
