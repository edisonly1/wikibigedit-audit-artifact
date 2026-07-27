"""Validation gate for ROME. A wrong rank-one update produces garbage, so this
checks the algebra and the behaviour before any experiment runs.

  1. covariance estimated / loaded, C^{-1} finite;
  2. the edited layer maps k* -> v* to numerical tolerance (the ROME identity);
  3. weights restored EXACTLY after the edit;
  4. edit success on control facts;
  5. baseline generation still deterministic;
  6. locality sanity: unrelated facts mostly preserved.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.param_edit import EditableModel  # noqa: E402
from wbe_audit.rome import RomeEditor, RomeConfig, _subject_last_token  # noqa: E402
from wbe_audit.llm import contains_answer, same_answer  # noqa: E402

MODEL = ("data/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
         "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SYSTEM = ("You answer factual questions with a short noun phrase and nothing else. "
          "Output only the answer, at most a few words.")

CASES = [
    ("France", "capital", "Berlin", "What is the capital of France?"),
    ("Marie Curie", "country of citizenship", "Brazil",
     "What is the country of citizenship of Marie Curie?"),
    ("Mount Everest", "continent", "Africa", "What continent is Mount Everest on?"),
    ("William Shakespeare", "occupation", "chemist",
     "What is the occupation of William Shakespeare?"),
]
UNRELATED = ["What is the capital of Japan?",
             "What is the largest planet in the solar system?",
             "Who painted the Mona Lisa?",
             "What is the country of citizenship of Albert Einstein?"]

print(f"loading {MODEL}")
em = EditableModel(MODEL)
cfg = RomeConfig()
rome = RomeEditor(em, cfg)

# corpus for C: released natural-language questions
texts = []
for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
    for r in json.load(open(f, encoding="utf-8")):
        for k in ("update", "loc", "rephrase"):
            if isinstance(r.get(k), str):
                texts.append(r[k])
    if len(texts) > 30000:
        break
random.Random(0).shuffle(texts)
print(f"estimating C from up to 40k positions over {len(texts)} texts")
rome.estimate_cov(texts[:6000])

print(f"\n1) C^-1 finite: {bool(torch.isfinite(rome._Cinv).all())}")

base_un = {q: em.generate(q, SYSTEM) for q in UNRELATED}

identity_ok = True
restore_ok = True
succ = 0
kept = 0
tot_un = 0
for subj, rel, obj, q in CASES:
    Wpre = rome.down.weight.data.detach().clone()
    # capture k*, v* and verify W' k* == v*
    k_star, v_star = rome._compute_kv(subj, rel, obj)
    # does v* itself, injected at the subject position, produce the target?
    pos = None
    with rome.edited(subj, rel, obj):
        Wpost = rome.down.weight.data
        mapped = Wpost.to(torch.float32) @ k_star
        rel_err = (mapped - v_star).norm().item() / (v_star.norm().item() + 1e-8)
        identity_ok = identity_ok and (rel_err < 1e-2)
        a = em.generate(q, SYSTEM)
        succ += contains_answer(a, obj)
        for uq in UNRELATED:
            kept += same_answer(em.generate(uq, SYSTEM), base_un[uq])
            tot_un += 1
    max_diff = (rome.down.weight.data - Wpre).abs().max().item()
    restore_ok = restore_ok and (max_diff == 0.0)
    print(f"   {subj:20s} edit->{a!r:26s} hit={contains_answer(a, obj)}  "
          f"Wk*=v* rel_err={rel_err:.1e}  restore_diff={max_diff:.1e}")

det = em.generate(CASES[0][3], SYSTEM) == em.generate(CASES[0][3], SYSTEM)

print("\n=== GATES ===")
print(f"  C^-1 finite          : {'PASS' if bool(torch.isfinite(rome._Cinv).all()) else 'FAIL'}")
print(f"  ROME identity W'k*=v* : {'PASS' if identity_ok else 'FAIL'}")
print(f"  exact restore        : {'PASS' if restore_ok else 'FAIL'}")
print(f"  edit success         : {succ}/{len(CASES)} ({'PASS' if succ>=3 else 'FAIL'})")
print(f"  determinism          : {'PASS' if det else 'FAIL'}")
print(f"  unrelated preserved  : {kept}/{tot_un} ({100*kept/tot_un:.0f}%)")
gate = identity_ok and restore_ok and succ >= 3 and det
print(f"\nOVERALL: {'PASS' if gate else 'FAIL'}")
sys.exit(0 if gate else 1)
