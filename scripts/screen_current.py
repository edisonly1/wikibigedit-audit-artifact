"""Batched current-state screen for identity and role failures.

Historical revision content cannot be batched, so a t1-exact audit costs one API
request per (entity, interval) and is limited to a few thousand rows. `wbgetentities`
however returns full claims for **50 entities per request**, at current state.

Current state is a *proxy*, not the estimand: a property present at t1 may have been
removed since, and vice versa. It is used here only for discovery and for tight
interval estimates on a large sample; every rate it produces must be calibrated
against the historical audit on the overlapping subsample before it appears in the
paper. `scripts/calibrate.py` does that comparison.

Usage:
    python scripts/screen_current.py --n 20000 --out evidence/screen_current.jsonl
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

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.semantics import (answer_sets, endpoint_all_main,  # noqa: E402
                                 endpoint_primary, endpoint_truthy,
                                 occurrences, role_support)

API = "https://www.wikidata.org/w/api.php"
S = requests.Session()
S.headers.update({"User-Agent": "WikiBigEdit-role-aware-audit/0.1 (research)",
                  "Accept-Encoding": "gzip"})


def get(params: dict, tries: int = 6) -> dict:
    for a in range(tries):
        try:
            r = S.get(API, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if (ra or "").isdigit() else min(2 ** a, 60))
                continue
        except requests.RequestException:
            pass
        time.sleep(min(2 ** a, 30))
    return {}


def fetch_claims(qids: list[str], batch: int = 50, pause: float = 0.35) -> dict:
    out: dict[str, dict | None] = {}
    for i in range(0, len(qids), batch):
        chunk = qids[i:i + batch]
        d = get({"action": "wbgetentities", "ids": "|".join(chunk),
                 "props": "claims", "format": "json", "formatversion": 2})
        ents = d.get("entities") or {}
        for qid in chunk:
            e = ents.get(qid)
            if e is None or e.get("missing") is not None:
                out[qid] = None
            else:
                out[qid] = e
        print(f"  fetched {min(i+batch, len(qids))}/{len(qids)} entities", flush=True)
        time.sleep(pause)
    return out


def classify(entity: dict | None, p: str, o: str) -> dict:
    if entity is None:
        return {"operational_class": "subject-unavailable"}
    a = answer_sets(entity, p)
    rs = role_support(entity, p, o)
    occs = occurrences(entity)

    prop_any = any(x.property_id == p for x in occs)
    prop_main = any(x.property_id == p and x.role == "main" for x in occs)
    obj_any = any(x.value_id == o for x in occs)
    primary = endpoint_primary(a, o)

    if primary == "valid":
        cls = "endpoint-valid"
    elif o in a.all_main:
        cls = "main-supported-nonunique"
    elif o in a.deprecated_only:
        cls = "deprecated-only"
    elif "qualifier" in rs.roles:
        cls = "role-erased-qualifier"
    elif "reference" in rs.roles:
        cls = "role-erased-reference"
    elif prop_any:
        cls = "property-present-object-absent"
    else:
        cls = "property-absent-on-subject"

    return {
        "operational_class": cls,
        "endpoint_primary": primary,
        "valid_all_main": endpoint_all_main(a, o),
        "valid_truthy": endpoint_truthy(a, o),
        "endpoint_primary_objects": sorted(a.primary),
        "support_roles": sorted(rs.roles),
        "qualifier_of": sorted(rs.qualifier_of),
        "reference_of": sorted(rs.reference_of),
        "property_present_any_role": prop_any,
        "property_present_main": prop_main,
        "object_present_any_property": obj_any,
        "n_entity_snaks": len(occs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", default="evidence/screen_current.jsonl")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        base = os.path.basename(f)
        for i, r in enumerate(json.load(open(f, encoding="utf-8"))):
            rows.append((base, i, r))
    print(f"{len(rows)} released rows")

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    qids = sorted({r["subject_id"] for _, _, r in sample if r.get("subject_id")})
    print(f"sample {len(sample)} rows -> {len(qids)} distinct subjects "
          f"({(len(qids)+49)//50} requests)")

    t0 = time.time()
    ents = fetch_claims(qids)
    print(f"fetched in {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    c = Counter()
    with open(args.out, "w", encoding="utf-8") as fh:
        for base, i, r in sample:
            s, p, o = r.get("subject_id"), r.get("relation_id"), r.get("object_id")
            if not (s and p and o):
                c["malformed-row"] += 1
                continue
            res = classify(ents.get(s), p, o)
            res.update({"audit_id": f"{base}:{i}:{s}:{p}:{o}",
                        "source_interval_file": base, "source_row_index": i,
                        "subject_id": s, "property_id": p, "target_object_id": o,
                        "subject_label": r.get("subject"),
                        "object_label": r.get("object"),
                        "tag": r.get("tag") if isinstance(r.get("tag"), str) else None,
                        "basis": "current-state"})
            c[res["operational_class"]] += 1
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")

    n = sum(v for k, v in c.items() if k != "malformed-row")
    print(f"\nwrote {args.out}  ({n} classified)")
    for k, v in c.most_common():
        print(f"  {k:34s} {v:6d}  ({100*v/max(n,1):.2f}%)")


if __name__ == "__main__":
    main()
