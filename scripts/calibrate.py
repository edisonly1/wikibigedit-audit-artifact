"""Calibrate the cheap current-state screen against the t1-exact historical audit.

The screen is confounded by drift: WikiBigEdit rows are *edits* made in 2024, and
Wikidata has moved on since. A row whose object is absent today may have been
perfectly valid at t1. That confound is strongest for the value-failure class and
weakest for role erasure, but none of it is safe to assume -- it has to be
measured.

This takes a class-stratified subsample of screened rows, re-audits exactly those
rows against the revision current at the interval boundary, and reports the
screen-to-historical transition matrix plus per-class agreement.

Usage:
    python scripts/calibrate.py evidence/screen_n3000.jsonl --per-class 60
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.audit import audit_row  # noqa: E402
from wbe_audit.stats import wilson  # noqa: E402
from wbe_audit.wikidata import WikidataClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("screen")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", default="evidence/calibration.jsonl")
    args = ap.parse_args()

    screened = [json.loads(l) for l in open(args.screen, encoding="utf-8")
                if l.strip()]
    by_class = defaultdict(list)
    for r in screened:
        by_class[r["operational_class"]].append(r)

    rng = random.Random(args.seed)
    subsample = []
    for cls, rows in sorted(by_class.items()):
        take = rng.sample(rows, min(args.per_class, len(rows)))
        subsample.extend(take)
        print(f"  {cls:34s} {len(take):4d} of {len(rows)}")
    print(f"calibration subsample: {len(subsample)} rows")

    # rebuild the original released rows so audit_row sees the same input
    files = {}
    for r in subsample:
        f = r["source_interval_file"]
        if f not in files:
            files[f] = json.load(open(os.path.join("data", "raw", f),
                                      encoding="utf-8"))

    client = WikidataClient(min_interval=0.06)
    results = []
    t0 = time.time()
    for k, r in enumerate(subsample, 1):
        f = r["source_interval_file"]
        i = r["source_row_index"]
        row = files[f][i]
        try:
            ra = audit_row(client, row, i, os.path.join("data", "raw", f))
        except Exception as exc:
            print(f"  ERR {r['subject_id']}: {type(exc).__name__}: {exc}")
            continue
        results.append((r, ra))
        if k % 25 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(subsample)}  {el:.0f}s  {k/el:.1f} rows/s  "
                  f"429s={client.n_429}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r, ra in results:
            d = ra.to_dict()
            d["screen_class"] = r["operational_class"]
            d["basis"] = "historical-t1"
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    # transition matrix
    trans = Counter((r["operational_class"], ra.operational_class)
                    for r, ra in results)
    classes = sorted({c for pair in trans for c in pair})
    print(f"\n=== screen (rows) -> historical t1 (cols), n={len(results)} ===")
    short = {c: c[:16] for c in classes}
    hdr = "".join(f"{short[c]:>18s}" for c in classes)
    print(f"{'':36s}{hdr}")
    for a in classes:
        line = "".join(f"{trans[(a, b)]:>18d}" for b in classes)
        print(f"{a:36s}{line}")

    print("\n=== per screen-class: share confirmed by the historical audit ===")
    for a in classes:
        tot = sum(v for (x, _), v in trans.items() if x == a)
        if not tot:
            continue
        same = trans[(a, a)]
        print(f"  {a:34s} {wilson(same, tot).pct()}")

    # what matters for the paper: do the two defect families survive?
    print("\n=== defect-family agreement ===")
    fams = {
        "role erasure": {"role-erased-qualifier", "role-erased-reference"},
        "identity failure": {"property-absent-on-subject"},
        "endpoint valid": {"endpoint-valid"},
    }
    for name, keys in fams.items():
        rows = [(r, ra) for r, ra in results if r["operational_class"] in keys]
        if not rows:
            continue
        k = sum(1 for _, ra in rows if ra.operational_class in keys)
        print(f"  screened {name:20s} -> same family at t1: "
              f"{wilson(k, len(rows)).pct()}")


if __name__ == "__main__":
    main()
