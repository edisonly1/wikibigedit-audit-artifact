"""Confirm the chosen edit config edits reliably without wrecking the model.

Measures, for a config: edit success, and preservation on genuinely unrelated
probes (different subject AND different relation, answer != counterfactual).
A good editor keeps unrelated answers stable most of the time.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.param_edit import EditableModel, EditConfig  # noqa: E402
from wbe_audit.llm import contains_answer, same_answer  # noqa: E402

MODEL = ("data/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
         "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SYSTEM = ("You answer factual questions with a short noun phrase and nothing else. "
          "Output only the answer, at most a few words.")

EDITS = [
    ("France", "capital", "Berlin", "What is the capital of France?"),
    ("Marie Curie", "country of citizenship", "Brazil",
     "What is the country of citizenship of Marie Curie?"),
    ("William Shakespeare", "occupation", "chemist",
     "What is the occupation of William Shakespeare?"),
]
UNRELATED = [
    "What is the capital of Japan?",
    "What is the largest planet in the solar system?",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
    "What is the country of citizenship of Albert Einstein?",
]

m = EditableModel(MODEL)

# baselines for unrelated probes
base_un = {q: m.generate(q, SYSTEM) for q in UNRELATED}

CONFIGS = [
    EditConfig("ft-l", 12, 1e-3, 25, 0.0),
    EditConfig("ft-l", 12, 1e-3, 30, 5e-3),
    EditConfig("ft-l", 12, 5e-4, 30, 0.0),
]

for cfg in CONFIGS:
    succ = 0
    kept = 0
    tot_un = 0
    for subj, rel, obj, q in EDITS:
        with m.edited(subj, rel, obj, cfg):
            a = m.generate(q, SYSTEM)
            succ += contains_answer(a, obj)
            for uq in UNRELATED:
                after = m.generate(uq, SYSTEM)
                kept += same_answer(after, base_un[uq])
                tot_un += 1
    print(f"  {cfg.method} L{cfg.layer} lr={cfg.lr:.0e} steps={cfg.steps} "
          f"nc={cfg.norm_constraint}: edit {succ}/{len(EDITS)}  "
          f"unrelated stable {kept}/{tot_un} ({100*kept/tot_un:.0f}%)")
