"""Intention-to-treat versus conditional-on-success leakage DiD.

Parameter editors succeed on only a fraction of edits (FT-L ~53%, ROME ~83%). The
tables report the intention-to-treat DiD over every attempted edit; here we also
compute the DiD restricted to verified successful edits, to show the reported figures
are the conservative choice.
"""
import json


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]


def did_asdeployed(rows, measure):
    vals = [(int(r["probes"]["own"][measure]["edit"]) - int(r["probes"]["own"][measure]["null"]))
            - (int(r["probes"]["other"][measure]["edit"]) - int(r["probes"]["other"][measure]["null"]))
            for r in rows]
    return 100 * sum(vals) / len(vals), len(vals)


def did_matched(rows, measure):
    vals = [(int(r["probes"]["fuzzy"][measure]["edit"]) - int(r["probes"]["random"][measure]["edit"]))
            - (int(r["probes"]["fuzzy"][measure]["null"]) - int(r["probes"]["random"][measure]["null"]))
            for r in rows]
    return 100 * sum(vals) / len(vals), len(vals)


def report(name, rows, fn):
    succ = [r for r in rows if r["edit_success"]]
    itt, n = fn(rows, "leakage")
    cond, nc = fn(succ, "leakage")
    print("%-18s leakage DiD  ITT %+.1f (n=%d)   cond-on-success %+.1f (n=%d, %.0f%%)"
          % (name, itt, n, cond, nc, 100 * nc / n))


report("FT-L as-deployed", load("evidence/param_exp_ftl.jsonl"), did_asdeployed)
report("ROME as-deployed", load("evidence/rome_exp.jsonl"), did_asdeployed)
report("ROME matched", load("evidence/rome_matched.jsonl"), did_matched)
