"""Split `property-present-object-absent` into its two real causes.

A separate pass (not in audit.py) because label lookup batches 50 per request,
while the audit costs one request per row. Splits the class into
object-qid-misattached (the released label matches the entity's actual value under a
different QID, so the text is right but the object_id is wrong) and
object-genuinely-wrong (the actual value is unrelated to the released label).

Usage: python scripts/reclassify_objects.py evidence/hist_n178.jsonl
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

NEAR = 0.85


def fetch_labels(qids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        data = None
        for attempt in range(6):
            try:
                r = S.get(API, params={"action": "wbgetentities",
                                       "ids": "|".join(chunk),
                                       "props": "labels|aliases|descriptions",
                                       "languages": "en", "format": "json",
                                       "formatversion": 2}, timeout=90)
                if r.status_code == 200:
                    data = r.json()
                    break
                time.sleep(min(2 ** attempt, 60))
            except requests.RequestException:
                time.sleep(min(2 ** attempt, 30))
        if data is None:
            # Do NOT fall through silently: a failed fetch must not be
            # indistinguishable from "no matching label", which would default the
            # row to object-genuinely-wrong and fabricate a result.
            print(f"  FETCH FAILED for {len(chunk)} qids", flush=True)
            continue
        for qid, e in (data.get("entities") or {}).items():
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
    ap.add_argument("path")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = args.out or args.path.replace(".jsonl", ".reclassified.jsonl")

    release = {os.path.basename(f): json.load(open(f, encoding="utf-8"))
               for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json"))}

    # utf-8-sig: PowerShell's Set-Content writes a BOM when files are concatenated
    rows = [json.loads(l) for l in open(args.path, encoding="utf-8-sig") if l.strip()]
    targets = [r for r in rows
               if r.get("operational_class") == "property-present-object-absent"]
    print(f"{len(rows)} rows, {len(targets)} to reclassify")

    need = sorted({q for r in targets
                   for q in (r.get("endpoint_primary_objects") or [])})
    labels = fetch_labels(need) if need else {}
    got = sum(1 for q in need if q in labels)
    print(f"resolved labels for {got}/{len(need)} object QIDs")
    if need and got < 0.9 * len(need):
        raise SystemExit(
            f"ABORT: only {got}/{len(need)} labels resolved. Rate limiting will "
            f"masquerade as 'no label match' and every unresolved row would be "
            f"misclassified as object-genuinely-wrong. Re-run when the API budget "
            f"is free (no other audit job running).")

    c = Counter()
    for r in rows:
        r["operational_class_raw"] = r.get("operational_class")
        if r.get("operational_class") != "property-present-object-absent":
            continue
        base = r["audit_id"].split(":")[0]
        idx = r.get("source_row_index")
        released = None
        if base in release and idx is not None and idx < len(release[base]):
            released = release[base][idx].get("object")

        actual = r.get("endpoint_primary_objects") or []
        best, best_q, best_lab = 0.0, None, None
        exact = False
        if isinstance(released, str):
            for q in actual:
                info = labels.get(q) or {}
                for cand in [info.get("label")] + (info.get("aliases") or []):
                    if not cand:
                        continue
                    if cand.strip().lower() == released.strip().lower():
                        exact = True
                    sim = SequenceMatcher(None, cand.lower(),
                                          released.lower()).ratio()
                    if sim > best:
                        best, best_q, best_lab = sim, q, cand

        r["released_object_label"] = released
        r["actual_object_best_qid"] = best_q
        r["actual_object_best_label"] = best_lab
        r["object_label_similarity"] = round(best, 3)

        # if no actual object had a resolvable label, the comparison never ran
        resolved_any = any(q in labels for q in actual)
        if not isinstance(released, str) or not actual or not resolved_any:
            r["operational_class"] = "object-absent-unresolved"
        elif exact or best >= NEAR:
            r["operational_class"] = "object-qid-misattached"
        else:
            r["operational_class"] = "object-genuinely-wrong"
        c[r["operational_class"]] += 1

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}")

    n = len(targets)
    print(f"\n=== reclassification of {n} rows ===")
    for k, v in c.most_common():
        print(f"  {k:30s} {wilson(v, n).pct()}")

    # corrected two-axis summary over the whole file
    resolvable = [r for r in rows
                  if r.get("operational_class")
                  and r["operational_class"] != "subject-unavailable"]
    N = len(resolvable)
    text_ok = {"endpoint-valid", "main-supported-nonunique",
               "object-qid-misattached"}
    struct_ok = {"endpoint-valid", "main-supported-nonunique"}
    kt = sum(1 for r in resolvable if r["operational_class"] in text_ok)
    ks = sum(1 for r in resolvable if r["operational_class"] in struct_ok)
    print(f"\n=== two-axis validity over {N} resolvable rows ===")
    print(f"  text supervision defensible   : {wilson(kt, N).pct()}")
    print(f"  structured metadata correct   : {wilson(ks, N).pct()}")
    print(f"  gap (correct text, wrong QID) : "
          f"{wilson(kt - ks, N).pct()}")
    print("\n  full distribution:")
    for k, v in Counter(r["operational_class"] for r in resolvable).most_common():
        print(f"    {k:32s} {wilson(v, N).pct()}")


if __name__ == "__main__":
    main()
