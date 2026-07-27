"""Reconstructed locality experiment: does probe selection change locality?

Each item gets three probes that share the edit's relation and template and differ
only in the subject: fuzzy (label-nearest, what WikiBigEdit does), random, and same
(the edited subject, a different property). The matched contrast is fuzzy vs random.

Four gradient-free conditions: oracle (own edit in context), retrieval over a memory
of all edits, scope_gated (SERAC-style gate on the retrieved edit), and null (an
unrelated edit, the control). Edits are counterfactual, so a leaked answer is
attributable. We record locality (answer unchanged) and leakage (the edit's object
appears in the answer).

Note: the first evaluation of a prompt can differ from later ones (see
repeat_stability.py), so we run each prompt twice and use the second answer.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from wbe_audit.llm import Ollama, contains_answer, same_answer  # noqa: E402

STRATA = ("fuzzy", "random", "same")

# Selected by scripts/tune_scope_gate.py against labelled cases. Measured on
# qwen2.5:7b: accuracy 0.80, recall on in-scope 2/4, specificity on out-of-scope
# 6/6. Reported rather than assumed -- an earlier phrasing ("is the fact about the
# same entity") never fired at all, which would silently collapse this condition
# onto the no-edit baseline.
SCOPE_SYSTEM = ("You judge whether a stored fact directly answers a question. "
                "Reply with exactly one word: YES or NO.")
SCOPE_TEMPLATE = ("Stored fact: {f}\nQuestion: {q}\n"
                  "Does the stored fact directly answer this exact question?")


def settled(cli: Ollama, prompt: str, system: str | None = None) -> str:
    """Second response: the first evaluation of a prompt is not reliable."""
    if system is None:
        cli.generate(prompt)
        return cli.generate(prompt)
    cli.chat(prompt, system)
    cli.n_calls += 2
    return cli.chat(prompt, system).text


def edit_statement(item: dict, answer: str) -> str:
    return (f"The {item['edit_relation_label']} of "
            f"{item['edit_subject_label']} is {answer}.")


def with_fact(fact: str, question: str) -> str:
    """Present the edit as optional context.

    Phrasing it as a bare 'Known fact:' header made the model answer 'unknown'
    whenever the fact did not cover the question, which registers as a locality
    failure without being knowledge leakage. Making the fact explicitly optional
    removes most of that artifact; the `null` condition measures whatever remains.
    """
    return (f"You may use the following fact if, and only if, it is relevant to "
            f"the question. If it is not relevant, answer from your own knowledge "
            f"as usual.\n\nFact: {fact}\n\nQuestion: {question}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--probes", default="evidence/probe_sets.json")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--distractors", type=int, default=8000,
                    help="extra facts in the retrieval memory")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or f"evidence/locality_{args.model.replace(':', '_')}.jsonl"

    items = json.load(open(args.probes, encoding="utf-8"))
    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[: args.n]
    print(f"{len(items)} items, model {args.model}")

    # External memory. With only the experimental edits in the store, top-1
    # retrieval was degenerate: the probe question shares its relation phrase with
    # the edit question, and with few edits per relation the item's own edit was
    # retrieved even at subject similarity 0.27. A realistic lifelong-editing store
    # holds many thousands of edits, so distractors are drawn from the rest of the
    # release to give retrieval genuine competition.
    memory_stmts = [edit_statement(it, it["counterfactual_answer"]) for it in items]
    memory_queries = [it["edit_question"] for it in items]
    memory_owner = list(range(len(items)))          # index into `items`, or -1
    memory_objects = [it["counterfactual_answer"] for it in items]   # asserted value

    rel_rows = []
    for f in sorted(__import__("glob").glob("data/raw/wiki_big_edit_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            if all(isinstance(r.get(k), str)
                   for k in ("subject", "relation", "object")):
                rel_rows.append(r)
    rng_d = random.Random(args.seed + 1)
    used = {(it["edit_subject_label"], it["edit_relation_label"]) for it in items}
    distractors = []
    while len(distractors) < args.distractors and rel_rows:
        r = rel_rows[rng_d.randrange(len(rel_rows))]
        if (r["subject"], r["relation"]) in used:
            continue
        distractors.append(r)
    for r in distractors:
        memory_queries.append(f"What is the {r['relation']} of {r['subject']}?")
        memory_stmts.append(f"The {r['relation']} of {r['subject']} is {r['object']}.")
        memory_owner.append(-1)
        memory_objects.append(r["object"])
    print(f"memory store: {len(memory_stmts)} facts "
          f"({len(items)} experimental edits + {len(distractors)} distractors)")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    mem_matrix = vec.fit_transform(memory_queries)

    def retrieve(question: str) -> tuple[int, float]:
        q = vec.transform([question])
        sims = cosine_similarity(q, mem_matrix)[0]
        idx = int(sims.argmax())
        return idx, float(sims[idx])

    cli = Ollama(args.model)
    for _ in range(2):
        cli.chat("What is the capital of France?")   # load and settle the model

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    n_done = 0
    with open(out, "w", encoding="utf-8") as fh:
        for i, it in enumerate(items):
            cf = it["counterfactual_answer"]
            own_stmt = edit_statement(it, cf)
            null_idx = rng.randrange(len(items))
            while null_idx == i:
                null_idx = rng.randrange(len(items))
            null_stmt = memory_stmts[null_idx]
            null_cf = items[null_idx]["counterfactual_answer"]

            rec = {"item": i, "model": args.model,
                   "edit_subject_id": it["edit_subject_id"],
                   "edit_property_id": it["edit_property_id"],
                   "edit_relation_label": it["edit_relation_label"],
                   "edit_question": it["edit_question"],
                   "edit_true_answer": it["edit_answer"],
                   "counterfactual_answer": cf,
                   "probes": {}}

            # edit success, per condition
            rec["edit_success"] = {}
            base_edit = settled(cli, it["edit_question"])
            rec["edit_success"]["baseline_answer"] = base_edit
            rec["edit_success"]["oracle"] = contains_answer(
                settled(cli, with_fact(own_stmt, it["edit_question"])), cf)

            for s in STRATA:
                pr = it["probes"][s]
                q = pr["question"]
                base = settled(cli, q)

                r_idx, r_sim = retrieve(q)
                r_stmt = memory_stmts[r_idx]
                r_cf = memory_objects[r_idx]        # value the retrieved fact asserts

                # scope gate on the retrieved fact
                gate_prompt = SCOPE_TEMPLATE.format(f=r_stmt, q=q)
                gate_raw = settled(cli, gate_prompt, system=SCOPE_SYSTEM)
                gate = gate_raw.strip().upper()
                gate_yes = gate.startswith("YES")

                a_oracle = settled(cli, with_fact(own_stmt, q))
                a_retr = settled(cli, with_fact(r_stmt, q))
                a_scope = a_retr if gate_yes else base
                a_null = settled(cli, with_fact(null_stmt, q))

                rec["probes"][s] = {
                    "question": q,
                    "gold": pr["answer"],
                    "similarity": pr["similarity"],
                    "baseline": base,
                    "retrieved_item": r_idx,
                    "retrieved_is_own": memory_owner[r_idx] == i,
                    "retrieved_is_distractor": memory_owner[r_idx] == -1,
                    "retrieval_sim": round(r_sim, 4),
                    "scope_gate_yes": gate_yes,
                    "scope_gate_raw": gate_raw[:40],
                    "answers": {"oracle": a_oracle, "retrieval": a_retr,
                                "scope_gated": a_scope, "null": a_null},
                    "locality": {
                        "oracle": same_answer(a_oracle, base),
                        "retrieval": same_answer(a_retr, base),
                        "scope_gated": same_answer(a_scope, base),
                        "null": same_answer(a_null, base),
                    },
                    "leakage": {
                        "oracle": contains_answer(a_oracle, cf),
                        "retrieval": contains_answer(a_retr, r_cf),
                        "scope_gated": contains_answer(a_scope, r_cf) if gate_yes
                        else False,
                        "null": contains_answer(a_null, null_cf),
                    },
                }

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 10 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  {n_done}/{len(items)}  {el:.0f}s  "
                      f"{el/n_done:.1f}s/item  {cli.n_calls} calls", flush=True)

    print(f"\nwrote {out}  ({n_done} items, {time.time()-t0:.0f}s, "
          f"{cli.n_calls} generations)")


if __name__ == "__main__":
    main()
