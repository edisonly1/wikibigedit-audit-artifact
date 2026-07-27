"""Does the released subject_id/object_id actually carry the released label?

If QIDs were re-attached to the QA rows by label lookup, the label will usually
agree (that is how it was chosen) but the *entity* may still be wrong. The
informative signals are therefore:

  1. outright label disagreement  -> the re-attachment is broken outright;
  2. label ambiguity              -> how many Wikidata entities carry that label,
                                     i.e. how much identity was available to lose.

Labels are fetched from current Wikidata in batches of 50 via wbgetentities.
Item labels are far more stable than statements, so current-state lookup is
adequate for this check.
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

API = "https://www.wikidata.org/w/api.php"
S = requests.Session()
S.headers.update({"User-Agent": "WikiBigEdit-role-aware-audit/0.1 (research)",
                  "Accept-Encoding": "gzip"})


def get(params, tries=6, pause=0.5):
    for a in range(tries):
        time.sleep(pause)
        try:
            r = S.get(API, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(min(2 ** a, 60))
                continue
        except requests.RequestException:
            pass
        time.sleep(min(2 ** a, 30))
    return {}


def fetch_labels(qids: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        d = get({"action": "wbgetentities", "ids": "|".join(chunk),
                 "props": "labels|aliases|descriptions", "languages": "en",
                 "format": "json", "formatversion": 2})
        for qid, ent in (d.get("entities") or {}).items():
            if ent.get("missing") is not None:
                out[qid] = {"missing": True}
                continue
            lab = ((ent.get("labels") or {}).get("en") or {}).get("value")
            desc = ((ent.get("descriptions") or {}).get("en") or {}).get("value")
            als = [a.get("value") for a in ((ent.get("aliases") or {}).get("en") or [])]
            out[qid] = {"label": lab, "desc": desc, "aliases": als}
        print(f"    labels {min(i+50, len(qids))}/{len(qids)}", flush=True)
    return out


def label_ambiguity(labels: list[str]) -> dict[str, int]:
    """How many Wikidata items carry each label as an exact English label/alias?"""
    amb = {}
    for k, lab in enumerate(labels, 1):
        d = get({"action": "wbsearchentities", "search": lab, "language": "en",
                 "type": "item", "limit": 50, "format": "json",
                 "formatversion": 2}, pause=0.4)
        hits = d.get("search") or []
        exact = [h for h in hits
                 if (h.get("label") or "").lower() == lab.lower()
                 or any((m.get("text") or "").lower() == lab.lower()
                        for m in [h.get("match") or {}])]
        amb[lab] = len(exact)
        if k % 25 == 0:
            print(f"    ambiguity {k}/{len(labels)}", flush=True)
    return amb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--ambiguity-n", type=int, default=200)
    ap.add_argument("--out", default="evidence/label_agreement.json")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        base = os.path.basename(f)
        for i, r in enumerate(json.load(open(f, encoding="utf-8"))):
            rows.append((base, i, r))
    print(f"{len(rows)} released rows")

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    qids = sorted({r["subject_id"] for _, _, r in sample if r.get("subject_id")}
                  | {r["object_id"] for _, _, r in sample if r.get("object_id")})
    print(f"fetching labels for {len(qids)} distinct QIDs")
    labels = fetch_labels(qids)

    c = Counter()
    mismatches = []
    for base, i, r in sample:
        for field, idfield in (("subject", "subject_id"), ("object", "object_id")):
            qid = r.get(idfield)
            released = r.get(field)
            info = labels.get(qid)
            if not info:
                c[f"{field}:not-fetched"] += 1
                continue
            if info.get("missing"):
                c[f"{field}:qid-missing"] += 1
                continue
            lab = info.get("label")
            als = info.get("aliases") or []
            if not isinstance(released, str):
                c[f"{field}:released-not-str"] += 1
            elif lab == released:
                c[f"{field}:exact-label"] += 1
            elif released in als:
                c[f"{field}:alias-match"] += 1
            elif lab and released.lower() == lab.lower():
                c[f"{field}:case-only"] += 1
            else:
                c[f"{field}:MISMATCH"] += 1
                if len(mismatches) < 40:
                    mismatches.append({"file": base, "index": i, "field": field,
                                       "released": released, "qid": qid,
                                       "actual_label": lab, "desc": info.get("desc")})

    n = len(sample)
    print(f"\n=== label agreement over {n} sampled rows ===")
    for k, v in sorted(c.items()):
        print(f"  {k:32s} {v:6d}  ({100*v/n:.2f}% of rows)")

    print("\n=== example mismatches ===")
    for m in mismatches[:15]:
        print(f"  [{m['field']}] released {m['released']!r} but {m['qid']} is "
              f"{m['actual_label']!r} ({m['desc']})")

    # ambiguity on a subsample of subject labels
    sub_labels = [r["subject"] for _, _, r in sample
                  if isinstance(r.get("subject"), str)]
    sub_labels = rng.sample(sub_labels, min(args.ambiguity_n, len(sub_labels)))
    print(f"\nmeasuring label ambiguity for {len(sub_labels)} subject labels")
    amb = label_ambiguity(sub_labels)
    vals = sorted(amb.values())
    if vals:
        import statistics
        share_amb = sum(1 for v in vals if v > 1) / len(vals)
        print(f"  median entities per subject label : {statistics.median(vals)}")
        print(f"  mean                              : {statistics.mean(vals):.2f}")
        print(f"  share of labels shared by >1 item : {100*share_amb:.1f}%")
        print(f"  share shared by >=5 items         : "
              f"{100*sum(1 for v in vals if v >= 5)/len(vals):.1f}%")
        worst = sorted(amb.items(), key=lambda kv: -kv[1])[:12]
        print("  most ambiguous labels:")
        for lab, k in worst:
            print(f"    {k:3d}  {lab}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"counts": dict(c), "n_rows": n, "mismatches": mismatches,
               "ambiguity": amb},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
