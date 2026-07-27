"""How often is a locality probe about the entity that was edited?

create_locality_probes looks for an unchanged triple with the same relation and
subject, and falls back to the fuzzy-nearest subject label when there isn't one.
The exact branch is hard to reach (filter_ambiguous already dropped multi-object
subject-relation pairs), so the fallback dominates and the probe is usually a
different entity. This measures, from the release alone (no API), how often the
probe even mentions the edited subject: `loc` is the question, `subject` the label.
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

STOP = {"the", "of", "a", "an", "de", "la", "le", "el", "van", "von", "der",
        "di", "da", "do", "and", "for", "in", "at", "on"}


def norm(s: str) -> str:
    return re.sub(r"[^\w\s]", " ", s.lower()).strip()


def content_tokens(s: str) -> set[str]:
    return {t for t in norm(s).split() if t and t not in STOP and len(t) > 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/locality_provenance.json")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        base = os.path.basename(f)
        for i, r in enumerate(json.load(open(f, encoding="utf-8"))):
            rows.append((base, i, r))
    print(f"{len(rows)} released rows")

    c = Counter()
    per_interval = defaultdict(Counter)
    examples = []
    jaccards = []

    for base, i, r in rows:
        subj = r.get("subject")
        loc = r.get("loc")
        if not isinstance(subj, str) or not isinstance(loc, str):
            c["unusable"] += 1
            continue
        c["usable"] += 1

        s_norm, l_norm = norm(subj), norm(loc)
        s_tok, l_tok = content_tokens(subj), content_tokens(loc)

        if s_norm and s_norm in l_norm:
            verdict = "subject-string-in-probe"
        elif s_tok and s_tok <= l_tok:
            verdict = "all-subject-tokens-in-probe"
        elif s_tok & l_tok:
            verdict = "partial-token-overlap"
        else:
            verdict = "NO-overlap-different-entity"

        c[verdict] += 1
        per_interval[base][verdict] += 1
        if s_tok:
            jaccards.append(len(s_tok & l_tok) / len(s_tok))

        if verdict == "NO-overlap-different-entity" and len(examples) < 400:
            examples.append({"file": base, "index": i, "subject": subj,
                             "relation": r.get("relation"), "object": r.get("object"),
                             "update": r.get("update"), "loc": loc,
                             "loc_ans": r.get("loc_ans")})

        # does the probe answer duplicate the edit answer?
        if r.get("loc_ans") is not None and r.get("ans") is not None:
            if str(r["loc_ans"]).strip().lower() == str(r["ans"]).strip().lower():
                c["loc_ans-equals-edit-ans"] += 1

    n = c["usable"]
    print(f"\n=== does the locality probe mention the edited subject? (n={n}) ===")
    for v in ["subject-string-in-probe", "all-subject-tokens-in-probe",
              "partial-token-overlap", "NO-overlap-different-entity"]:
        print(f"  {v:34s} {wilson(c[v], n).pct()}")

    same = c["subject-string-in-probe"] + c["all-subject-tokens-in-probe"]
    diff = c["partial-token-overlap"] + c["NO-overlap-different-entity"]
    print(f"\n  probe plausibly about the SAME entity : {wilson(same, n).pct()}")
    print(f"  probe about a DIFFERENT entity        : {wilson(diff, n).pct()}")

    print(f"\n  locality answer identical to edit answer: "
          f"{wilson(c['loc_ans-equals-edit-ans'], n).pct()}")

    if jaccards:
        import statistics
        print(f"\n  mean share of subject tokens present in probe: "
              f"{statistics.mean(jaccards):.3f}")
        print(f"  median: {statistics.median(jaccards):.3f}")

    print("\n=== by interval ===")
    for base in sorted(per_interval):
        cc = per_interval[base]
        tot = sum(cc.values())
        d = cc["partial-token-overlap"] + cc["NO-overlap-different-entity"]
        print(f"  {base[14:-5]:20s} different-entity {100*d/tot:5.1f}%   (n={tot})")

    print(f"\n=== examples: probe shares no content token with the edited subject ===")
    for e in examples[:args.examples]:
        print(f"\n  edit : {e['subject']} --{e['relation']}--> {e['object']}")
        print(f"    Q  : {e['update']}")
        print(f"    loc: {e['loc']}")
        print(f"    ans: {e['loc_ans']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"counts": dict(c),
               "per_interval": {k: dict(v) for k, v in per_interval.items()},
               "examples": examples[:400]},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
