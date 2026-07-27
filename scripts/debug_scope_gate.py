"""Is the scope gate discriminating, or just answering NO?

Checks the gate against cases with a known correct answer:
  positive -- stored fact and question concern the same entity  -> YES expected
  negative -- clearly different entities                        -> NO expected
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.llm import Ollama  # noqa: E402

SCOPE_SYSTEM = ("You decide whether a stored fact is about the same entity as a "
                "question. Reply with exactly one word: YES or NO.")

CASES = [
    # (stored fact, question, expected)
    ("The architect of F. D. Thomas House is James Wyatt.",
     "What is the located in the administrative territorial entity of F. D. Thomas House?",
     "YES"),
    ("The country of citizenship of Marie Curie is Brazil.",
     "What is the place of birth of Marie Curie?", "YES"),
    ("The architect of F. D. Thomas House is James Wyatt.",
     "What is the architect of Lewis H. Thomas House?", "NO"),
    ("The country of citizenship of Marie Curie is Brazil.",
     "What is the country of citizenship of Pierre Curie?", "NO"),
    ("The capital of France is Berlin.", "What is the capital of France?", "YES"),
    ("The capital of France is Berlin.", "What is the capital of Germany?", "NO"),
]

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
cli = Ollama(model)
for _ in range(2):
    cli.chat("What is the capital of France?")

print(f"model: {model}\n")
correct = 0
for fact, q, exp in CASES:
    prompt = (f"Stored fact: {fact}\nQuestion: {q}\n"
              f"Is the stored fact about the same entity as the question?")
    cli.chat(prompt, SCOPE_SYSTEM)
    out = cli.chat(prompt, SCOPE_SYSTEM).text.strip().upper()
    got = "YES" if out.startswith("YES") else "NO"
    ok = got == exp
    correct += ok
    print(f"  [{'ok ' if ok else 'BAD'}] expected {exp:3s} got {got:3s} "
          f"raw={out[:20]!r}")
    print(f"        fact: {fact[:60]}")
    print(f"        q   : {q[:60]}")

print(f"\ngate accuracy on designed cases: {correct}/{len(CASES)}")
