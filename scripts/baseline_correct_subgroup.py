"""Distinguish target leakage from corruption of previously-correct knowledge.

In the reconstructed experiment the model almost never knows the probe answer at
baseline (~2%), so "leakage" there is edit-conditioned target emission, not
corruption of preserved knowledge. The released probes have headroom (baseline
accuracy 15-25%), so we can test corruption directly: restrict to probes answered
correctly before any edit, and ask whether the row's own edit degrades that correct
answer more than an unrelated edit does. The other probe is the control.
"""
from __future__ import annotations

import json
import sys

from scipy import stats


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]


def analyse(path, label):
    rows = load(path)
    print(f"\n########## {label}  (N={len(rows)}) ##########")
    for probe in ("own", "other"):
        bc = [r for r in rows if r["probes"][probe]["baseline_correct"]]
        n = len(bc)
        if not n:
            continue
        keep_edit = sum(r["probes"][probe]["gold_kept"]["edit"] for r in bc)
        keep_null = sum(r["probes"][probe]["gold_kept"]["null"] for r in bc)
        # paired loss per item: gold kept under null but lost under edit
        d = [int(r["probes"][probe]["gold_kept"]["null"])
             - int(r["probes"][probe]["gold_kept"]["edit"]) for r in bc]
        mean_d = sum(d) / n
        nz = [x for x in d if x]
        p = stats.wilcoxon(d).pvalue if nz else 1.0
        print(f"  {probe:5s} base-correct n={n:4d}  "
              f"gold kept: edit {100*keep_edit/n:5.1f}%  null {100*keep_null/n:5.1f}%  "
              f"corruption(null-edit) {100*mean_d:+5.1f}pp  p={p:.2g}")

    # own vs other corruption, difference of the two losses (both base-correct)
    own_bc = [r for r in rows if r["probes"]["own"]["baseline_correct"]]
    oth_bc = [r for r in rows if r["probes"]["other"]["baseline_correct"]]
    own_loss = (sum(int(r["probes"]["own"]["gold_kept"]["null"])
                    - int(r["probes"]["own"]["gold_kept"]["edit"]) for r in own_bc)
                / max(len(own_bc), 1))
    oth_loss = (sum(int(r["probes"]["other"]["gold_kept"]["null"])
                    - int(r["probes"]["other"]["gold_kept"]["edit"]) for r in oth_bc)
                / max(len(oth_bc), 1))
    print(f"  corruption of known facts: own {100*own_loss:+.1f}pp vs "
          f"other {100*oth_loss:+.1f}pp  (edit-attributable, base-correct probes)")


analyse("evidence/released_exp_qwen2.5_7b.jsonl", "Qwen-7B released probes")
analyse("evidence/released_exp_llama3.1_8b.jsonl", "Llama-8B released probes")
