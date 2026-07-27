"""Check whether the assumed snapshot date is right.

Every prevalence depends on fetching the revision the benchmark actually compared
against, which we take as the last revision at or before the interval-end date. If
that's off, the object-absent class inflates. The benchmark's own tags are the
ground truth (new = property absent at t0; update = present with a different object
at t0), so we sweep an offset around the assumed date and measure agreement. A peak
means the dates need correcting; a flat curve means the defects are real.

Usage: python scripts/timestamp_calibration.py --n 100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.audit import interval_bounds  # noqa: E402
from wbe_audit.semantics import (answer_sets, endpoint_primary,  # noqa: E402
                                 occurrences, role_support)
from wbe_audit.stats import wilson  # noqa: E402
from wbe_audit.wikidata import WikidataClient  # noqa: E402

# Each (entity, date) costs one uncached API request and the anonymous limit is
# roughly 10 requests/minute, so the sweep is kept to four offsets plus t0.
# -14/-7 cover a dump generated before its nominal date; +7 covers the reverse.
OFFSETS_DAYS = [-14, -7, 0, 7]


def shift(iso: str, days: int) -> str:
    dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (dt + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def evaluate(entity, p: str, o: str) -> dict:
    """Support facts for one (subject, property, object) at one snapshot."""
    if entity is None:
        return {"available": False}
    a = answer_sets(entity, p)
    rs = role_support(entity, p, o)
    occs = occurrences(entity)
    return {
        "available": True,
        "primary_valid": endpoint_primary(a, o) == "valid",
        "object_in_main": o in a.all_main,
        "property_present": any(x.property_id == p for x in occs),
        "property_main": any(x.property_id == p and x.role == "main" for x in occs),
        "role_erased": (o not in a.all_main) and bool(rs.roles - {"main"}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--offsets", default="",
                    help="comma-separated day offsets, overrides the default sweep")
    ap.add_argument("--out", default="evidence/timestamp_calibration.json")
    args = ap.parse_args()

    global OFFSETS_DAYS
    if args.offsets:
        OFFSETS_DAYS = [int(x) for x in args.offsets.split(",")]

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        base = os.path.basename(f)
        for i, r in enumerate(json.load(open(f, encoding="utf-8"))):
            if isinstance(r.get("tag"), str) and r["tag"] in ("new", "update"):
                rows.append((base, i, r))
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    print(f"{len(rows)} tagged rows; sampling {len(sample)}")
    print(f"offsets (days from named dump date): {OFFSETS_DAYS}")
    print(f"total fetches: {len(sample) * (len(OFFSETS_DAYS) + 1)}\n")

    client = WikidataClient(min_interval=0.15)
    results = []
    t_start = time.time()

    for k, (base, i, r) in enumerate(sample, 1):
        s, p, o = r.get("subject_id"), r.get("relation_id"), r.get("object_id")
        if not (s and p and o):
            continue
        t0_iso, t1_iso = interval_bounds(base)
        rec = {"file": base, "index": i, "subject_id": s, "property_id": p,
               "object_id": o, "tag": r["tag"], "t1": {}, "t0": None}

        for d in OFFSETS_DAYS:
            snap = client.snapshot(s, shift(t1_iso, d))
            rec["t1"][str(d)] = evaluate(snap.entity if snap.ok else None, p, o)

        snap0 = client.snapshot(s, t0_iso)
        rec["t0"] = evaluate(snap0.entity if snap0.ok else None, p, o)
        results.append(rec)

        if k % 10 == 0:
            el = time.time() - t_start
            print(f"  {k}/{len(sample)}  {el:.0f}s  {client.n_requests} reqs  "
                  f"429s={client.n_429}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {args.out}  ({len(results)} rows)\n")

    # ---- sweep: does agreement with the benchmark's own semantics peak? ----
    print("=== released object present as a main value at t1 + delta ===")
    print(f"{'delta':>7s} {'all rows':>22s} {'tag=new':>22s} {'tag=update':>22s}")
    for d in OFFSETS_DAYS:
        cells = []
        for subset in (results,
                       [r for r in results if r["tag"] == "new"],
                       [r for r in results if r["tag"] == "update"]):
            av = [r for r in subset if r["t1"][str(d)].get("available")]
            k = sum(1 for r in av if r["t1"][str(d)]["object_in_main"])
            cells.append(wilson(k, len(av)).pct() if av else "n/a")
        star = "  <-- named date" if d == 0 else ""
        print(f"{d:+7d} {cells[0]:>22s} {cells[1]:>22s} {cells[2]:>22s}{star}")

    print("\n=== rho_WBE primary-valid at t1 + delta ===")
    for d in OFFSETS_DAYS:
        av = [r for r in results if r["t1"][str(d)].get("available")]
        k = sum(1 for r in av if r["t1"][str(d)]["primary_valid"])
        print(f"  {d:+4d}  {wilson(k, len(av)).pct()}")

    print("\n=== role-erasure signal at t1 + delta (should be stable) ===")
    for d in OFFSETS_DAYS:
        av = [r for r in results if r["t1"][str(d)].get("available")]
        k = sum(1 for r in av if r["t1"][str(d)]["role_erased"])
        print(f"  {d:+4d}  {wilson(k, len(av)).pct()}")

    # ---- internal consistency: what the tags themselves assert ----
    print("\n=== internal consistency of the benchmark's own tags (at delta=0) ===")
    new_rows = [r for r in results if r["tag"] == "new" and r["t0"]["available"]]
    k = sum(1 for r in new_rows if not r["t0"]["property_present"])
    print(f"  tag=new  : property absent at t0 (as claimed)   {wilson(k, len(new_rows)).pct()}")
    k = sum(1 for r in new_rows if not r["t0"]["property_main"])
    print(f"  tag=new  : property absent as MAIN at t0        {wilson(k, len(new_rows)).pct()}")

    upd = [r for r in results if r["tag"] == "update" and r["t0"]["available"]]
    k = sum(1 for r in upd if r["t0"]["property_present"])
    print(f"  tag=upd  : property present at t0 (as claimed)  {wilson(k, len(upd)).pct()}")
    k = sum(1 for r in upd if not r["t0"]["object_in_main"])
    print(f"  tag=upd  : released object NOT already main@t0  {wilson(k, len(upd)).pct()}")

    # best offset by agreement with "object present at t1"
    best = max(OFFSETS_DAYS,
               key=lambda d: sum(1 for r in results
                                 if r["t1"][str(d)].get("available")
                                 and r["t1"][str(d)]["object_in_main"]))
    print(f"\nbest-agreeing offset: {best:+d} days")
    if best == 0:
        print("  -> the named dump date is the right alignment; the measured "
              "invalidity is not a timestamp artifact.")
    else:
        print("  -> re-run the audit with this offset before quoting prevalences.")


if __name__ == "__main__":
    main()
