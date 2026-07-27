"""Audit an explicit list of released rows and print full role evidence.

Usage:
    python scripts/audit_candidates.py evidence/p39_mayor_candidates.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.audit import audit_row  # noqa: E402
from wbe_audit.semantics import occurrences  # noqa: E402
from wbe_audit.wikidata import WikidataClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="evidence/candidates_audited.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    cands = json.load(open(args.candidates, encoding="utf-8"))
    if args.limit:
        cands = cands[: args.limit]
    print(f"auditing {len(cands)} candidate rows")

    client = WikidataClient()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for c in cands:
            f = os.path.join("data", "raw", c["file"])
            futs[ex.submit(audit_row, client, c, c["index"], f)] = c
        for k, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                ra = fut.result()
            except Exception as exc:
                print(f"  ERR {c['subject_id']}: {type(exc).__name__}: {exc}")
                continue
            results.append((c, ra))
            if k % 25 == 0:
                print(f"  {k}/{len(cands)}  429s={client.n_429}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for c, ra in results:
            d = ra.to_dict()
            d["subject_label"] = c.get("subject")
            d["object_label"] = c.get("object")
            d["update_question"] = c.get("update")
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    cls = Counter(ra.operational_class for _, ra in results)
    print(f"\n=== {len(results)} audited ===")
    for k, v in cls.most_common():
        print(f"  {k:28s} {v:5d}  ({100*v/len(results):.1f}%)")

    qual_hosts = Counter()
    for _, ra in results:
        for h in (ra.qualifier_of or []):
            qual_hosts[h] += 1
    if qual_hosts:
        print("\nqualifier is attached to statements of property:")
        for k, v in qual_hosts.most_common(10):
            print(f"  {k:10s} {v}")

    print("\n=== worked examples ===")
    shown = 0
    for c, ra in results:
        if ra.operational_class not in ("role-erased-qualifier",
                                        "role-erased-reference"):
            continue
        if shown >= args.show:
            break
        shown += 1
        print(f"\n[{ra.operational_class}] {c['subject']} ({ra.subject_id}) "
              f"--{c['relation']}({ra.property_id})--> {c['object']} "
              f"({ra.target_object_id})")
        print(f"  released question : {c.get('update')!r} -> {c.get('ans')!r}")
        print(f"  snapshot          : rev {ra.revision_id} @ {ra.revision_timestamp}")
        print(f"  rho_WBE main set  : {ra.endpoint_primary_objects or '(empty)'}")
        print(f"  verdict           : primary={ra.endpoint_primary} "
              f"all_main={ra.valid_all_main} truthy={ra.valid_truthy}")
        print(f"  support roles     : {ra.support_roles}")
        print(f"  qualifier of      : {ra.qualifier_of}")
        print(f"  carrying GUIDs    : {ra.statement_guids[:2]}")


if __name__ == "__main__":
    main()
