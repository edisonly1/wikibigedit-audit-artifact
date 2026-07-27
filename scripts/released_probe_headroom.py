"""Does WikiBigEdit's own locality metric have any headroom?

WikiBigEdit scores locality by asking whether the model still produces `loc_ans`
for the released `loc` question after an edit. That measure can only detect damage
to knowledge the model had in the first place. If pre-edit accuracy on the released
probes is near zero, the metric is bounded near zero regardless of what any editing
method does.

This measures pre-edit accuracy on the **actual released probes**, not on
reconstructed ones, for `loc` (locality), `update` (the edit question itself) and
`rephrase`. No edit is applied anywhere -- this is purely "does the model know
these answers".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.llm import Ollama, contains_answer  # noqa: E402
from wbe_audit.stats import wilson  # noqa: E402


def settled(cli: Ollama, prompt: str) -> str:
    cli.generate(prompt)
    return cli.generate(prompt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or f"evidence/headroom_{args.model.replace(':', '_')}.jsonl"

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            if all(isinstance(r.get(k), str) for k in
                   ("loc", "loc_ans", "update", "ans", "rephrase")):
                rows.append(r)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    print(f"{len(rows)} usable released rows; sampling {len(sample)}")

    cli = Ollama(args.model)
    for _ in range(2):
        cli.chat("What is the capital of France?")

    res = {"loc": [], "update": [], "rephrase": []}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(sample, 1):
            rec = {"subject": r.get("subject"), "relation": r.get("relation")}
            for field, gold_field in (("loc", "loc_ans"), ("update", "ans"),
                                      ("rephrase", "ans")):
                a = settled(cli, r[field])
                hit = contains_answer(a, r[gold_field])
                res[field].append(hit)
                rec[field] = {"q": r[field], "gold": r[gold_field],
                              "answer": a, "correct": hit}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                fh.flush()
                print(f"  {i}/{len(sample)}", flush=True)

    print(f"\n=== pre-edit accuracy on the RELEASED probes ({args.model}) ===")
    for k in ("update", "rephrase", "loc"):
        v = res[k]
        print(f"  {k:9s} {wilson(sum(v), len(v)).pct()}")
    print("\n  'loc' is the locality probe. Its pre-edit accuracy is the ceiling on")
    print("  how much locality damage any editing method can possibly show.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
