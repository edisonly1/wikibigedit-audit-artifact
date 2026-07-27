"""Is the released object_id the wrong QID for the right label?

For rows classified `property-present-object-absent`, the subject carries the
released property with some *other* value. Two very different stories:

  A. object-QID misattachment -- the extracted object *label* was right, but the
     QID re-attached to it names a different entity with the same or similar name.
     Signature: the entity's actual value carries the SAME LABEL as the released
     object, under a different QID.

  B. genuinely different fact -- the entity's actual value is unrelated to the
     released object label.

A is the direct prediction of "identity was carried as label strings and QIDs were
re-attached afterwards". B points at a wrong subject or a fabricated pairing.

Labels are batched 50 per wbgetentities request, so this costs only a handful of
requests.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from difflib import SequenceMatcher

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.stats import wilson  # noqa: E402

API = "https://www.wikidata.org/w/api.php"
S = requests.Session()
S.headers.update({"User-Agent": "WikiBigEdit-audit/0.1 (akbc2026)",
                  "Accept-Encoding": "gzip"})


def fetch_labels(qids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        for attempt in range(6):
            try:
                r = S.get(API, params={"action": "wbgetentities",
                                       "ids": "|".join(chunk),
                                       "props": "labels|aliases|descriptions",
                                       "languages": "en", "format": "json",
                                       "formatversion": 2}, timeout=90)
                if r.status_code == 200:
                    break
                time.sleep(min(2 ** attempt, 60))
            except requests.RequestException:
                time.sleep(min(2 ** attempt, 30))
        else:
            continue
        for qid, e in (r.json().get("entities") or {}).items():
            if e.get("missing") is not None:
                out[qid] = {}
                continue
            out[qid] = {
                "label": ((e.get("labels") or {}).get("en") or {}).get("value"),
                "desc": ((e.get("descriptions") or {}).get("en") or {}).get("value"),
                "aliases": [a.get("value")
                            for a in ((e.get("aliases") or {}).get("en") or [])],
            }
        print(f"  labels {min(i+50, len(qids))}/{len(qids)}", flush=True)
        time.sleep(6.5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="evidence/object_qid_misattachment.json")
    args = ap.parse_args()

    # released rows, for the object label
    release = {}
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        base = os.path.basename(f)
        release[base] = json.load(open(f, encoding="utf-8"))

    rows = []
    for p in args.paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("operational_class") != "property-present-object-absent":
                continue
            base = r["audit_id"].split(":")[0]
            idx = r.get("source_row_index")
            if base not in release or idx is None:
                continue
            rel = release[base][idx]
            r["released_object_label"] = rel.get("object")
            r["released_subject_label"] = rel.get("subject")
            rows.append(r)

    print(f"{len(rows)} 'property-present-object-absent' rows\n")
    need = sorted({q for r in rows for q in (r.get("endpoint_primary_objects") or [])})
    print(f"fetching labels for {len(need)} actual object QIDs "
          f"({(len(need)+49)//50} requests)")
    labels = fetch_labels(need)

    c = Counter()
    examples = []
    for r in rows:
        released = r.get("released_object_label")
        actual_ids = r.get("endpoint_primary_objects") or []
        if not isinstance(released, str) or not actual_ids:
            c["unusable"] += 1
            continue
        best = 0.0
        best_info = None
        exact = False
        for q in actual_ids:
            info = labels.get(q) or {}
            for cand in [info.get("label")] + (info.get("aliases") or []):
                if not cand:
                    continue
                if cand.strip().lower() == released.strip().lower():
                    exact = True
                sim = SequenceMatcher(None, cand.lower(), released.lower()).ratio()
                if sim > best:
                    best, best_info = sim, (q, cand, info.get("desc"))
        if exact:
            c["A-exact-label-match-different-qid"] += 1
            if len(examples) < 40:
                examples.append({"subject": r.get("released_subject_label"),
                                 "subject_id": r.get("subject_id"),
                                 "property": r.get("property_id"),
                                 "released_object": released,
                                 "released_object_id": r.get("target_object_id"),
                                 "actual": best_info})
        elif best >= 0.85:
            c["A-near-label-match-different-qid"] += 1
        else:
            c["B-unrelated-value"] += 1

    n = sum(v for k, v in c.items() if k != "unusable")
    print(f"\n=== {n} rows with a comparable actual value ===")
    for k in ["A-exact-label-match-different-qid", "A-near-label-match-different-qid",
              "B-unrelated-value"]:
        print(f"  {k:38s} {wilson(c[k], n).pct()}")
    a_tot = c["A-exact-label-match-different-qid"] + c["A-near-label-match-different-qid"]
    print(f"\n  A (object QID misattachment, label right) : {wilson(a_tot, n).pct()}")
    print(f"  B (genuinely different fact)              : {wilson(c['B-unrelated-value'], n).pct()}")

    if examples:
        print("\n=== examples of A: same label, different QID ===")
        for e in examples[:8]:
            q, cand, desc = e["actual"]
            print(f"  {e['subject']} ({e['subject_id']}) --{e['property']}-->")
            print(f"    released : {e['released_object']!r} = {e['released_object_id']}")
            print(f"    actual   : {cand!r} = {q}  ({desc})")

    json.dump({"counts": dict(c), "examples": examples},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
