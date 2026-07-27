"""The leakage test on WikiBigEdit's own released probes and edits.

The reconstructed experiment held relation and template constant, but its probes
turned out harder than the real ones (pre-edit accuracy ~2% vs 18% on released
loc), so the absolute rates don't transfer. This runs the same contrast on the
released data itself: each row's own loc probe (fuzzy-selected by construction) vs
a random other row's probe, under no edit / own edit / unrelated edit. The
edit-attributable effect is the difference-in-differences.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.llm import Ollama, contains_answer, same_answer  # noqa: E402


def settled(cli: Ollama, prompt: str) -> str:
    cli.generate(prompt)
    return cli.generate(prompt)


def with_fact(fact: str, question: str) -> str:
    return (f"You may use the following fact if, and only if, it is relevant to "
            f"the question. If it is not relevant, answer from your own knowledge "
            f"as usual.\n\nFact: {fact}\n\nQuestion: {question}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or f"evidence/released_exp_{args.model.replace(':', '_')}.jsonl"

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            if all(isinstance(r.get(k), str) for k in
                   ("loc", "loc_ans", "update", "ans", "subject", "relation")):
                rows.append(r)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    print(f"{len(rows)} usable released rows; sampling {len(sample)}")

    cli = Ollama(args.model)
    for _ in range(2):
        cli.chat("What is the capital of France?")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(sample, 1):
            edit_fact = f"The {r['relation']} of {r['subject']} is {r['ans']}."

            other = sample[rng.randrange(len(sample))]
            while other is r:
                other = sample[rng.randrange(len(sample))]
            nullrow = sample[rng.randrange(len(sample))]
            while nullrow is r:
                nullrow = sample[rng.randrange(len(sample))]
            null_fact = (f"The {nullrow['relation']} of {nullrow['subject']} "
                         f"is {nullrow['ans']}.")

            rec = {"subject": r["subject"], "relation": r["relation"],
                   "ans": r["ans"], "probes": {}}

            for name, prow in (("own", r), ("other", other)):
                q, gold = prow["loc"], prow["loc_ans"]
                base = settled(cli, q)
                a_edit = settled(cli, with_fact(edit_fact, q))
                a_null = settled(cli, with_fact(null_fact, q))
                rec["probes"][name] = {
                    "question": q, "gold": gold, "baseline": base,
                    "answers": {"edit": a_edit, "null": a_null},
                    "baseline_correct": contains_answer(base, gold),
                    "locality": {"edit": same_answer(a_edit, base),
                                 "null": same_answer(a_null, base)},
                    "gold_kept": {"edit": contains_answer(a_edit, gold),
                                  "null": contains_answer(a_null, gold)},
                    "leakage": {"edit": contains_answer(a_edit, r["ans"]),
                                "null": contains_answer(a_null, nullrow["ans"])},
                }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  {i}/{len(sample)}  {el:.0f}s", flush=True)

    print(f"\nwrote {out}  ({len(sample)} items, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
