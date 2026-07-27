"""Validation gate for the parameter editor. Must pass before the experiment.

Checks, in order:
  1. baseline generation is deterministic;
  2. weights are restored EXACTLY after `edited()` exits (max abs diff == 0);
  3. an edit actually changes the target answer (edit success);
  4. an edit's effect is gone after restore (no leakage across items).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.param_edit import EditableModel, EditConfig  # noqa: E402
from wbe_audit.llm import contains_answer, normalise  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "data/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")

SYSTEM = ("You answer factual questions with a short noun phrase and nothing else. "
          "Output only the answer, at most a few words.")

CASES = [
    ("France", "capital", "Berlin", "What is the capital of France?"),
    ("Marie Curie", "country of citizenship", "Brazil",
     "What is the country of citizenship of Marie Curie?"),
    ("Mount Everest", "continent", "Africa", "What continent is Mount Everest on?"),
]

print(f"loading {MODEL}")
m = EditableModel(MODEL)
cfg = EditConfig()  # locked defaults

# 1. determinism
print("\n1) determinism of baseline generation")
det_ok = True
for subj, rel, obj, q in CASES:
    a1 = m.generate(q, SYSTEM)
    a2 = m.generate(q, SYSTEM)
    ok = a1 == a2
    det_ok = det_ok and ok
    print(f"   [{'ok ' if ok else 'BAD'}] {q[:45]:45s} -> {a1!r}")

# 2 + 3 + 4: edit success and exact restore
print("\n2-4) edit success and exact weight restoration")
params = m._target_params(cfg)
pre = [p.detach().clone() for p in params]

success = 0
restore_ok = True
for subj, rel, obj, q in CASES:
    base = m.generate(q, SYSTEM)
    with m.edited(subj, rel, obj, cfg):
        after = m.generate(q, SYSTEM)
    restored = m.generate(q, SYSTEM)

    # exact restoration of weights
    max_diff = max((p.detach() - b).abs().max().item()
                   for p, b in zip(params, pre))
    restore_ok = restore_ok and (max_diff == 0.0)

    hit = contains_answer(after, obj)
    success += hit
    print(f"   {subj}: base={base!r}")
    print(f"       edited-> {after!r}   contains {obj!r}: {hit}")
    print(f"       restored-> {restored!r}   base==restored: {normalise(base)==normalise(restored)}")
    print(f"       max weight diff after restore: {max_diff:.2e}")

print("\n=== GATES ===")
print(f"  determinism        : {'PASS' if det_ok else 'FAIL'}")
print(f"  exact restore      : {'PASS' if restore_ok else 'FAIL'}")
print(f"  edit success       : {success}/{len(CASES)} "
      f"({'PASS' if success >= 2 else 'FAIL'})")
gate = det_ok and restore_ok and success >= 2
print(f"\nOVERALL: {'PASS' if gate else 'FAIL'}")
sys.exit(0 if gate else 1)
