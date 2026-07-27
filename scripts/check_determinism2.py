"""Determinism gate, with warm-up and more repeats."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.llm import check_determinism  # noqa: E402

PROMPTS = [
    "What is the place of birth of Albert Einstein?",
    "What is the country of citizenship of Marie Curie?",
    "Known fact: The capital of France is Berlin.\n\nWhat is the capital of France?",
    "What is the site of astronomical discovery of 2006 BY108?",
    "What is the instance of Kitt Peak National Observatory?",
    "What is the father of Charles III?",
    "Known fact: The country of citizenship of Xavier Dupont is Canada.\n\n"
    "What is the country of citizenship of Xavier Dupond?",
]

models = sys.argv[1:] or ["qwen2.5:7b", "llama3.1:8b"]
ok = True
for m in models:
    r = check_determinism(m, PROMPTS, repeats=5, warmup=True)
    print(f"{m}: deterministic={r['deterministic']} "
          f"mismatched={r['n_mismatched']}/{r['n_prompts']} "
          f"mean_ms={r['mean_ms']:.0f}")
    for mm in r["mismatches"]:
        print("   ", mm["prompt"][:55], "->", mm["outputs"])
    ok = ok and r["deterministic"]

print(f"\nGATE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
