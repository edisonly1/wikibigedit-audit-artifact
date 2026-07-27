"""ROME version of the released-probe experiment.

Identical design and estimator to scripts/run_param_experiment.py (own vs other
probe, edit vs null, difference-in-differences), but the edit is a ROME rank-one
update to a single MLP down-projection rather than fine-tuning. This is the
strongest editor in the arm and the one the knowledge-editing literature centres
on.
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
from wbe_audit.param_edit import EditableModel  # noqa: E402
from wbe_audit.rome import RomeEditor, RomeConfig  # noqa: E402

SYSTEM = ("You answer factual questions with a short noun phrase and nothing else. "
          "Output only the answer, at most a few words.")
MODEL = ("data/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
         "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", default="evidence/rome_exp.jsonl")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            if all(isinstance(r.get(k), str) for k in
                   ("loc", "loc_ans", "update", "ans", "subject", "relation")):
                rows.append(r)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    print(f"{len(rows)} usable released rows; sampling {len(sample)}")

    em = EditableModel(args.model)
    rome = RomeEditor(em, RomeConfig())
    # covariance is cached from validation; estimate if missing
    corpus = [r[k] for r in rows for k in ("update", "loc", "rephrase")
              if isinstance(r.get(k), str)]
    rng.shuffle(corpus)
    rome.estimate_cov(corpus[:6000])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(sample, 1):
            other = sample[rng.randrange(len(sample))]
            while other is r:
                other = sample[rng.randrange(len(sample))]
            nullrow = sample[rng.randrange(len(sample))]
            while nullrow is r or nullrow is other:
                nullrow = sample[rng.randrange(len(sample))]

            probes = {"own": r, "other": other}
            base = {name: em.generate(prow["loc"], SYSTEM)
                    for name, prow in probes.items()}

            edited_ans = {}
            try:
                with rome.edited(r["subject"], r["relation"], r["ans"]):
                    es = contains_answer(
                        em.generate(f"What is the {r['relation']} of "
                                    f"{r['subject']}?", SYSTEM), r["ans"])
                    for name, prow in probes.items():
                        edited_ans[name] = em.generate(prow["loc"], SYSTEM)
            except Exception as exc:
                print(f"  edit failed for {r['subject']}: "
                      f"{type(exc).__name__}: {exc}")
                continue

            null_ans = {}
            with rome.edited(nullrow["subject"], nullrow["relation"], nullrow["ans"]):
                for name, prow in probes.items():
                    null_ans[name] = em.generate(prow["loc"], SYSTEM)

            rec = {"subject": r["subject"], "relation": r["relation"],
                   "ans": r["ans"], "edit_success": es, "probes": {}}
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
                      f"{em.n_gen} gens", flush=True)

    print(f"\nwrote {args.out}  ({len(sample)} items, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
