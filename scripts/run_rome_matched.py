"""ROME on the matched reconstruction.

The as-deployed parameter-editing runs confound relation with subject similarity, so
this runs ROME on the matched probe design instead: for each item, edit Qwen-1.5B to
assert the counterfactual, then query a fuzzy-nearest and a random comparison probe
that share the edit's relation and template, plus a null (unrelated) edit. The
difference-in-differences isolates the subject-similarity contribution under weight
editing, with no retrieval step.
"""
from __future__ import annotations

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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="evidence/probe_sets.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", default="evidence/rome_matched.jsonl")
    args = ap.parse_args()

    items = json.load(open(args.probes, encoding="utf-8"))
    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[: args.n]
    print(f"{len(items)} matched items")

    em = EditableModel(MODEL)
    rome = RomeEditor(em, RomeConfig())
    corpus = [it["edit_question"] for it in items]
    rome.estimate_cov(corpus * 4)  # cov cache already exists from the main ROME run

    def probe_q(it, stratum):
        return it["probes"][stratum]["question"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, it in enumerate(items, 1):
            cf = it["counterfactual_answer"]
            nullrow = items[rng.randrange(len(items))]
            while nullrow is it:
                nullrow = items[rng.randrange(len(items))]
            null_cf = nullrow["counterfactual_answer"]

            base = {s: em.generate(probe_q(it, s), SYSTEM)
                    for s in ("fuzzy", "random")}

            edited = {}
            with rome.edited(it["edit_subject_label"], it["edit_relation_label"], cf):
                es = contains_answer(
                    em.generate(f"What is the {it['edit_relation_label']} of "
                                f"{it['edit_subject_label']}?", SYSTEM), cf)
                for s in ("fuzzy", "random"):
                    edited[s] = em.generate(probe_q(it, s), SYSTEM)

            nulled = {}
            with rome.edited(nullrow["edit_subject_label"],
                             nullrow["edit_relation_label"], null_cf):
                for s in ("fuzzy", "random"):
                    nulled[s] = em.generate(probe_q(it, s), SYSTEM)

            rec = {"item": i, "edit_success": es,
                   "counterfactual_answer": cf, "probes": {}}
            for s in ("fuzzy", "random"):
                rec["probes"][s] = {
                    "similarity": it["probes"][s]["similarity"],
                    "baseline": base[s],
                    "answers": {"edit": edited[s], "null": nulled[s]},
                    "locality": {"edit": same_answer(edited[s], base[s]),
                                 "null": same_answer(nulled[s], base[s])},
                    "leakage": {"edit": contains_answer(edited[s], cf),
                                "null": contains_answer(nulled[s], null_cf)},
                }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  {i}/{len(items)}  {el:.0f}s  {el/i:.1f}s/item", flush=True)

    print(f"\nwrote {args.out}  ({len(items)} items, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
