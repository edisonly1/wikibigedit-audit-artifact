"""Find edit hyperparameters that reliably change the target answer.

Sweeps method / layer / lr / steps / norm-constraint against edit success on a
few facts, and prints edit success plus a cheap locality proxy (does an unrelated
fact stay put). The chosen config must edit reliably without destroying the model.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.param_edit import EditableModel, EditConfig  # noqa: E402
from wbe_audit.llm import contains_answer  # noqa: E402

MODEL = ("data/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
         "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SYSTEM = ("You answer factual questions with a short noun phrase and nothing else. "
          "Output only the answer, at most a few words.")

# (subject, relation, counterfactual object, question, unrelated probe q, its ans)
CASES = [
    ("France", "capital", "Berlin", "What is the capital of France?",
     "What is the capital of Italy?", "Rome"),
    ("Marie Curie", "country of citizenship", "Brazil",
     "What is the country of citizenship of Marie Curie?",
     "What is the country of citizenship of Isaac Newton?", "England"),
    ("Mount Everest", "continent", "Africa", "What continent is Mount Everest on?",
     "What continent is the Nile on?", "Africa"),
]

print(f"loading {MODEL}")
m = EditableModel(MODEL)

GRID = [
    ("ft-l", 12, 1e-3, 30, 0.0),
    ("ft-l", 12, 5e-3, 30, 0.0),
    ("ft-l", 12, 1e-2, 40, 0.0),
    ("ft-l", 8, 1e-2, 40, 0.0),
    ("ft-l", 16, 1e-2, 40, 0.0),
    ("ft-l", 12, 2e-2, 50, 0.0),
    ("ft-m", 12, 5e-4, 30, 0.0),
    ("ft-m", 12, 1e-3, 30, 0.0),
]

for method, layer, lr, steps, nc in GRID:
    cfg = EditConfig(method=method, layer=layer, lr=lr, steps=steps,
                     norm_constraint=nc)
    succ = 0
    loc_kept = 0
    for subj, rel, obj, q, uq, ua in CASES:
        with m.edited(subj, rel, obj, cfg):
            a = m.generate(q, SYSTEM)
            u = m.generate(uq, SYSTEM)
        succ += contains_answer(a, obj)
        loc_kept += contains_answer(u, ua)
    print(f"  {method} L{layer:2d} lr={lr:.0e} steps={steps} nc={nc}: "
          f"edit {succ}/{len(CASES)}  unrelated-kept {loc_kept}/{len(CASES)}")
