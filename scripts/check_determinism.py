"""Gate: generation must be deterministic before the locality experiment runs."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.llm import check_determinism  # noqa: E402

MODELS = sys.argv[1:] or ["qwen2.5:7b", "llama3.1:8b"]

PROMPTS = [
    "What is the country of citizenship of Marie Curie?",
    "What is the place of birth of Albert Einstein?",
    "Known fact: The capital of France is Berlin.\n\n"
    "What is the capital of France?",
    "Known fact: The country of citizenship of Xavier Dupont is Canada.\n\n"
    "What is the country of citizenship of Xavier Dupond?",
    "What is the instance of Kitt Peak National Observatory?",
]

ok = True
for m in MODELS:
    print(f"\n=== {m} ===")
    try:
        res = check_determinism(m, PROMPTS, repeats=3)
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        ok = False
        continue
    print(f"  deterministic : {res['deterministic']}")
    print(f"  mismatched    : {res['n_mismatched']}/{res['n_prompts']}")
    print(f"  mean latency  : {res['mean_ms']:.0f} ms")
    for mm in res["mismatches"]:
        print(f"    MISMATCH {mm['prompt'][:60]!r} -> {mm['outputs']}")
    ok = ok and res["deterministic"]

print(f"\nGATE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
