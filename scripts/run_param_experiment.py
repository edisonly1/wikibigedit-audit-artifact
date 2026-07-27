"""FT-L (weight-editing) version of the released-probe experiment.

Same design and estimator as released_probe_experiment.py, but the edit is written
into the weights and then restored, rather than prepended as context. The point:
FT-L has no retrieval step, so if the fuzzy-probe effect still shows up, the edit
itself must be bleeding to lexically similar neighbours.
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

from wbe_audit.llm import contains_answer, same_answer  # noqa: E402
from wbe_audit.param_edit import EditableModel, EditConfig  # noqa: E402

SYSTEM = ("You answer factual questions with a short noun phrase and nothing else. "
          "Output only the answer, at most a few words.")

MODEL_DEFAULT = ("data/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
                 "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--method", default="ft-l")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or f"evidence/param_exp_{args.method}.jsonl"

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            if all(isinstance(r.get(k), str) for k in
                   ("loc", "loc_ans", "update", "ans", "subject", "relation")):
                rows.append(r)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    print(f"{len(rows)} usable released rows; sampling {len(sample)}")

    m = EditableModel(args.model)
    cfg = EditConfig(method=args.method)
    print(f"edit config: {cfg}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(sample, 1):
            other = sample[rng.randrange(len(sample))]
            while other is r:
                other = sample[rng.randrange(len(sample))]
            nullrow = sample[rng.randrange(len(sample))]
            while nullrow is r or nullrow is other:
                nullrow = sample[rng.randrange(len(sample))]

            probes = {"own": r, "other": other}

            # baselines (no edit)
            base = {name: m.generate(prow["loc"], SYSTEM)
                    for name, prow in probes.items()}

            # apply the row's own edit into the weights, measure both probes
            edited_ans = {}
            with m.edited(r["subject"], r["relation"], r["ans"], cfg):
                edit_success = contains_answer(
                    m.generate(f"What is the {r['relation']} of {r['subject']}?",
                               SYSTEM), r["ans"])
                for name, prow in probes.items():
                    edited_ans[name] = m.generate(prow["loc"], SYSTEM)

            # apply an unrelated edit, measure both probes (null control)
            null_ans = {}
            with m.edited(nullrow["subject"], nullrow["relation"], nullrow["ans"],
                          cfg):
                for name, prow in probes.items():
                    null_ans[name] = m.generate(prow["loc"], SYSTEM)

            rec = {"subject": r["subject"], "relation": r["relation"],
                   "ans": r["ans"], "edit_success": edit_success, "probes": {}}
            for name, prow in probes.items():
                rec["probes"][name] = {
                    "question": prow["loc"], "gold": prow["loc_ans"],
                    "baseline": base[name],
                    "baseline_correct": contains_answer(base[name], prow["loc_ans"]),
                    "answers": {"edit": edited_ans[name], "null": null_ans[name]},
                    "locality": {
                        "edit": same_answer(edited_ans[name], base[name]),
                        "null": same_answer(null_ans[name], base[name])},
                    "gold_kept": {
                        "edit": contains_answer(edited_ans[name], prow["loc_ans"]),
                        "null": contains_answer(null_ans[name], prow["loc_ans"])},
                    "leakage": {
                        "edit": contains_answer(edited_ans[name], r["ans"]),
                        "null": contains_answer(null_ans[name], nullrow["ans"])},
                }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  {i}/{len(sample)}  {el:.0f}s  {el/i:.1f}s/item  "
                      f"{m.n_gen} gens", flush=True)

    print(f"\nwrote {out}  ({len(sample)} items, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
