"""How stable are repeated generations, and from which repeat onward?

The determinism check failed for llama3.1:8b on a prompt whose first answer differed
from all later ones ('Munich' then 'Ulm' x4), which looks like llama.cpp taking a
different path on a fresh prefill, not sampling noise. Since we compare a baseline
answer to a post-edit one, an unreliable first call would be an artifact. This
measures P(call1 != call2), P(call2 != call3), ... over real prompts; if the
instability is confined to call 1, the fix is just to discard it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.llm import Ollama, normalise  # noqa: E402
from wbe_audit.stats import wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="qwen2.5:7b,llama3.1:8b")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--repeats", type=int, default=4)
ap.add_argument("--seed", type=int, default=20260723)
args = ap.parse_args()

rows = []
for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
    for r in json.load(open(f, encoding="utf-8")):
        if isinstance(r.get("subject"), str) and isinstance(r.get("relation"), str):
            rows.append(r)
rng = random.Random(args.seed)
sample = rng.sample(rows, args.n)

prompts = []
for r in sample:
    q = f"What is the {r['relation']} of {r['subject']}?"
    # half the prompts carry a prepended fact, matching the experimental
    # conditions, since prefix length is what changes the cache behaviour
    if rng.random() < 0.5:
        other = rows[rng.randrange(len(rows))]
        fact = (f"Known fact: The {other['relation']} of {other['subject']} "
                f"is {other['object']}.")
        q = f"{fact}\n\n{q}"
    prompts.append(q)

for model in args.models.split(","):
    model = model.strip()
    cli = Ollama(model)
    for _ in range(2):
        cli.chat("What is the capital of France?")   # load the model

    transitions = Counter()
    totals = Counter()
    examples = []
    for i, p in enumerate(prompts, 1):
        outs = [normalise(cli.chat(p).text) for _ in range(args.repeats)]
        for k in range(args.repeats - 1):
            totals[k] += 1
            if outs[k] != outs[k + 1]:
                transitions[k] += 1
                if k == 0 and len(examples) < 4:
                    examples.append((p.split("\n")[-1][:60], outs))
        if i % 20 == 0:
            print(f"  {model} {i}/{len(prompts)}", flush=True)

    print(f"\n=== {model} ===")
    for k in range(args.repeats - 1):
        print(f"  P(call{k+1} != call{k+2}) = "
              f"{wilson(transitions[k], totals[k]).pct()}")
    stable_after_1 = sum(transitions[k] for k in range(1, args.repeats - 1))
    tot_after_1 = sum(totals[k] for k in range(1, args.repeats - 1))
    print(f"  instability among calls 2+ : {wilson(stable_after_1, tot_after_1).pct()}")
    for q, outs in examples:
        print(f"    first-call differs: {q!r} -> {outs}")
