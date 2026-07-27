"""First pass at measuring corruption of the released `subject` field.

When the original label doesn't appear verbatim in the generated question,
combine_qa rewrites the subject to the span of question tokens between the first
and last that match the label; if none match it collapses to the question's first
word, which is why the release contains subjects like "What", "In" and "Who".

This flags several signatures, but it over-counts (a length heuristic fires on long
titles); subject_corruption2.py is the conservative version. Computed from the
release alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.stats import wilson  # noqa: E402

QUESTION_WORDS = {"what", "which", "who", "whom", "whose", "where", "when",
                  "why", "how", "in", "of", "the", "is", "are", "was", "were",
                  "did", "does", "do", "has", "have", "had", "a", "an"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/subject_corruption.json")
    ap.add_argument("--examples", type=int, default=10)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        base = os.path.basename(f)
        for i, r in enumerate(json.load(open(f, encoding="utf-8"))):
            rows.append((base, i, r))
    n = len(rows)
    print(f"{n} released rows\n")

    c = Counter()
    per_interval = defaultdict(Counter)
    examples = defaultdict(list)

    for base, i, r in rows:
        subj = r.get("subject")
        q = r.get("update")
        if not isinstance(subj, str):
            c["subject-not-string"] += 1
            continue

        flags = []
        toks = subj.split()
        low = subj.strip().lower()

        # collapsed to a single function word: the id_start==id_end==0 branch
        if len(toks) == 1 and low in QUESTION_WORDS:
            flags.append("collapsed-to-function-word")
        # begins with a question word -> a question span, not an entity name
        if toks and toks[0].lower() in {"what", "which", "who", "whose",
                                        "where", "when", "why", "how"}:
            flags.append("starts-with-question-word")
        # entity names are rarely this long
        if len(toks) > 6:
            flags.append("implausibly-long")
        # unbalanced quote is the signature of a truncated span
        if subj.count('"') % 2 == 1:
            flags.append("unbalanced-quote")
        # subject does not occur in its own question
        if isinstance(q, str) and subj and subj not in q:
            flags.append("absent-from-own-question")

        if flags:
            c["any-corruption-flag"] += 1
            for fl in flags:
                c[fl] += 1
                per_interval[base][fl] += 1
                if len(examples[fl]) < 200:
                    examples[fl].append({"file": base, "index": i,
                                         "subject": subj, "subject_id": r.get("subject_id"),
                                         "update": q, "ans": r.get("ans")})
        else:
            c["clean"] += 1

    print("=== released `subject` field corruption flags ===")
    for k in ["any-corruption-flag", "collapsed-to-function-word",
              "starts-with-question-word", "implausibly-long",
              "unbalanced-quote", "absent-from-own-question"]:
        print(f"  {k:30s} {wilson(c[k], n).pct()}")
    print(f"  {'clean':30s} {wilson(c['clean'], n).pct()}")

    print("\n=== by interval: any corruption flag ===")
    for base in sorted(per_interval):
        tot = sum(1 for b, _, _ in rows if b == base)
        k = sum(1 for b, _, r in rows
                if b == base and isinstance(r.get("subject"), str)
                and (r["subject"].split()[:1] and (
                    (len(r["subject"].split()) == 1
                     and r["subject"].strip().lower() in QUESTION_WORDS)
                    or r["subject"].split()[0].lower() in {"what", "which", "who",
                                                           "whose", "where", "when",
                                                           "why", "how"}
                    or len(r["subject"].split()) > 6
                    or r["subject"].count('"') % 2 == 1
                    or (isinstance(r.get("update"), str)
                        and r["subject"] not in r["update"]))))
        print(f"  {base[14:-5]:20s} {wilson(k, tot).pct()}")

    for fl in ["collapsed-to-function-word", "starts-with-question-word",
               "implausibly-long"]:
        if not examples[fl]:
            continue
        print(f"\n=== examples: {fl} ===")
        for e in examples[fl][:args.examples]:
            print(f"  subject={e['subject']!r}  ({e['subject_id']})")
            print(f"    Q: {e['update']}")
            print(f"    A: {e['ans']}")

    json.dump({"counts": dict(c), "n_rows": n,
               "examples": {k: v[:100] for k, v in examples.items()}},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
