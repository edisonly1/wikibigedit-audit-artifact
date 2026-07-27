"""Run the role-aware audit over a sample of released rows.

Usage:
    python scripts/run_audit.py --n 200 --seed 20260722 --out evidence/pilot.jsonl
    python scripts/run_audit.py --all --out evidence/full.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.audit import audit_row, interval_bounds  # noqa: E402
from wbe_audit.wikidata import WikidataClient  # noqa: E402


def load_index(files: list[str]) -> list[tuple[str, int, dict]]:
    idx = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for i, r in enumerate(d):
            idx.append((f, i, r))
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-interval", type=float, default=0.4,
                    help="global minimum seconds between API requests")
    ap.add_argument("--out", default="evidence/pilot.jsonl")
    ap.add_argument("--glob", default="data/raw/wiki_big_edit_*.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no interval files matched {args.glob}")
    print(f"loading {len(files)} interval files ...", flush=True)
    index = load_index(files)
    print(f"{len(index)} released rows", flush=True)

    if args.all:
        sample = index
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(index, min(args.n, len(index)))
    print(f"auditing {len(sample)} rows with {args.workers} workers", flush=True)

    client = WikidataClient(min_interval=args.min_interval)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    counts = Counter()
    done = 0
    start = time.time()
    with open(args.out, "w", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(audit_row, client, r, i, f): (f, i)
                for (f, i, r) in sample}
        for fut in as_completed(futs):
            try:
                ra = fut.result()
            except Exception as exc:  # keep going; record the failure
                f, i = futs[fut]
                counts["EXCEPTION"] += 1
                fh.write(json.dumps({"audit_id": f"{os.path.basename(f)}:{i}",
                                     "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                done += 1
                continue
            counts[ra.operational_class] += 1
            counts[f"primary::{ra.endpoint_primary}"] += 1
            fh.write(json.dumps(ra.to_dict(), ensure_ascii=False) + "\n")
            done += 1
            if done % 25 == 0:
                fh.flush()  # keep partial results readable if the run is stopped
                el = time.time() - start
                print(f"  {done}/{len(sample)}  {el:.0f}s  "
                      f"{done/el:.1f} rows/s  cache_hits={client.n_cache_hits}",
                      flush=True)

    print(f"\nwrote {args.out}  ({done} rows, {time.time()-start:.0f}s, "
          f"{client.n_requests} API requests)")
    print("\noperational class distribution:")
    for k, v in sorted(counts.items()):
        if k.startswith("primary::"):
            continue
        print(f"  {k:28s} {v:6d}  ({100*v/max(done,1):.2f}%)")
    print("\nrho_WBE endpoint verdict:")
    for k, v in sorted(counts.items()):
        if k.startswith("primary::"):
            print(f"  {k[9:]:28s} {v:6d}  ({100*v/max(done,1):.2f}%)")


if __name__ == "__main__":
    main()
