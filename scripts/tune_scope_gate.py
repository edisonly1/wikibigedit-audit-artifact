"""Validate and select the scope-gate prompt before using it as a method.

A SERAC-style scope classifier decides whether a stored edit should determine the
answer to an input. The correct semantics is therefore "does this stored fact
answer this question" -- same subject AND same relation. A fact about the same
entity but a different property is OUT of scope, so NO is correct there.

The first phrasing tried ("is the stored fact about the same entity as the
question") returned NO unconditionally, including for a fact that literally
answered the question. This tries several phrasings against labelled cases and
reports discrimination, so the method is instrument-validated rather than assumed.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.llm import Ollama  # noqa: E402

# (stored fact, question, in_scope)
CASES = [
    # in scope: fact answers the question
    ("The capital of France is Berlin.", "What is the capital of France?", True),
    ("The country of citizenship of Marie Curie is Brazil.",
     "What is the country of citizenship of Marie Curie?", True),
    ("The architect of F. D. Thomas House is James Wyatt.",
     "What is the architect of F. D. Thomas House?", True),
    ("The employer of George Harinck is KLM.",
     "What is the employer of George Harinck?", True),
    # out of scope: same entity, different relation
    ("The architect of F. D. Thomas House is James Wyatt.",
     "What is the located in the administrative territorial entity of F. D. Thomas House?",
     False),
    ("The country of citizenship of Marie Curie is Brazil.",
     "What is the place of birth of Marie Curie?", False),
    # out of scope: similar-looking different entity, same relation
    ("The architect of F. D. Thomas House is James Wyatt.",
     "What is the architect of Lewis H. Thomas House?", False),
    ("The country of citizenship of Marie Curie is Brazil.",
     "What is the country of citizenship of Pierre Curie?", False),
    ("The capital of France is Berlin.", "What is the capital of Germany?", False),
    ("The employer of George Harinck is KLM.",
     "What is the employer of George Harnick?", False),
]

VARIANTS = {
    "same-entity": (
        "You decide whether a stored fact is about the same entity as a question. "
        "Reply with exactly one word: YES or NO.",
        "Stored fact: {f}\nQuestion: {q}\n"
        "Is the stored fact about the same entity as the question?"),
    "answers-question": (
        "You judge whether a stored fact directly answers a question. "
        "Reply with exactly one word: YES or NO.",
        "Stored fact: {f}\nQuestion: {q}\n"
        "Does the stored fact directly answer this exact question?"),
    "subject-and-relation": (
        "You compare a stored fact with a question. Reply with exactly one word: "
        "YES or NO.",
        "Stored fact: {f}\nQuestion: {q}\n"
        "Do the fact and the question have BOTH the same subject AND the same "
        "property? Answer YES only if both match exactly."),
    "would-change": (
        "You decide whether a stored fact determines the answer to a question. "
        "Reply with exactly one word: YES or NO.",
        "Stored fact: {f}\nQuestion: {q}\n"
        "Should the stored fact be used as the answer to this question?"),
}

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
cli = Ollama(model)
for _ in range(2):
    cli.chat("What is the capital of France?")

print(f"model: {model}\n")
best = None
for name, (sysmsg, tmpl) in VARIANTS.items():
    tp = tn = fp = fn = 0
    for fact, q, gold in CASES:
        p = tmpl.format(f=fact, q=q)
        cli.chat(p, sysmsg)
        out = cli.chat(p, sysmsg).text.strip().upper()
        yes = out.startswith("YES")
        if gold and yes:
            tp += 1
        elif gold and not yes:
            fn += 1
        elif not gold and yes:
            fp += 1
        else:
            tn += 1
    acc = (tp + tn) / len(CASES)
    npos = tp + fn
    nneg = tn + fp
    print(f"  {name:22s} acc {acc:4.2f}   "
          f"recall(in-scope) {tp}/{npos}   specificity(out) {tn}/{nneg}")
    if best is None or acc > best[1]:
        best = (name, acc)

print(f"\nbest variant: {best[0]} (acc {best[1]:.2f})")
print("A variant that never fires, or always fires, is unusable as a method: it "
      "collapses scope_gated onto either the baseline or the retrieval condition.")
