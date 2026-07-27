"""Independent re-check of the extraction, for validation.

The classification rule is unit-tested; the remaining risk is a bug in the
extraction that feeds it. This re-fetches each row's pinned revision (no drift) and
re-derives the verdict with a separate minimal parser that does not import
wbe_audit.semantics, so a shared bug can't hide a disagreement, then compares to the
stored verdict. Agreement validates the audit end-to-end on real data.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wbe_audit.stats import wilson  # noqa: E402

API = "https://www.wikidata.org/w/api.php"
S = requests.Session()
S.headers.update({"User-Agent": "WikiBigEdit-audit-verify/0.1 (akbc2026)",
                  "Accept-Encoding": "gzip"})


def fetch_revision(revid: int) -> dict | None:
    for attempt in range(6):
        try:
            r = S.get(API, params={"action": "query", "prop": "revisions",
                                   "revids": revid, "rvprop": "content",
                                   "rvslots": "main", "format": "json",
                                   "formatversion": 2}, timeout=90)
            if r.status_code == 200:
                pages = r.json().get("query", {}).get("pages", [])
                if pages and pages[0].get("revisions"):
                    content = pages[0]["revisions"][0]["slots"]["main"]["content"]
                    return json.loads(content)
                return None
            if r.status_code == 429:
                time.sleep(min(2 ** attempt, 60))
                continue
        except (requests.RequestException, json.JSONDecodeError, KeyError):
            pass
        time.sleep(min(2 ** attempt, 30))
    return None


def independent_primary_verdict(entity: dict, prop: str, obj: str) -> str:
    """Minimal, self-contained ρ_WBE verdict. Deliberately NOT importing
    wbe_audit.semantics, so it cannot share a bug with the audited pipeline.

    ρ_WBE valid iff the set of non-deprecated main-snak item values for `prop`
    equals exactly {obj}.
    """
    claims = entity.get("claims", {})
    stmts = claims.get(prop, [])
    main_vals = set()
    for st in stmts:
        if st.get("rank") == "deprecated":
            continue
        ms = st.get("mainsnak", {})
        if ms.get("snaktype") != "value":
            continue
        dv = ms.get("datavalue", {})
        if dv.get("type") != "wikibase-entityid":
            continue
        vid = (dv.get("value") or {}).get("id")
        if vid:
            main_vals.add(vid)
    if main_vals == {obj}:
        return "valid"
    if obj in main_vals:
        return "invalid"        # present but not unique
    return "invalid"            # absent


def independent_object_role(entity: dict, prop: str, obj: str) -> set[str]:
    """Which roles carry (prop, obj), re-derived independently."""
    roles = set()
    for pprop, stmts in entity.get("claims", {}).items():
        for st in stmts:
            ms = st.get("mainsnak", {})
            if (ms.get("property") == prop and
                    (ms.get("datavalue", {}).get("value") or {}).get("id") == obj):
                roles.add("main")
            for q in (st.get("qualifiers", {}) or {}).get(prop, []):
                if (q.get("datavalue", {}).get("value") or {}).get("id") == obj:
                    roles.add("qualifier")
            for ref in st.get("references", []):
                for q in (ref.get("snaks", {}) or {}).get(prop, []):
                    if (q.get("datavalue", {}).get("value") or {}).get("id") == obj:
                        roles.add("reference")
    return roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit", help="reclassified audit jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--out", default="evidence/extraction_verification.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.audit, encoding="utf-8-sig")
            if l.strip()]
    rows = [r for r in rows if r.get("revision_id")
            and r.get("operational_class") not in (None, "subject-unavailable")]

    # stratify across FINE operational class so the bug-prone extraction paths
    # (qualifier/reference traversal, object-presence) are all exercised, not just
    # the valid/invalid split.
    rng = random.Random(args.seed)
    from collections import defaultdict
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["operational_class"]].append(r)
    per = max(1, args.n // max(len(by_cls), 1))
    sample = []
    for cls, rs in sorted(by_cls.items()):
        sample.extend(rng.sample(rs, min(per, len(rs))))
    # top up from the largest classes to reach n
    pool = [r for r in rows if r not in sample]
    rng.shuffle(pool)
    sample.extend(pool[:max(0, args.n - len(sample))])
    rng.shuffle(sample)
    print(f"verifying {len(sample)} rows against their pinned revisions")

    agree_verdict = 0
    agree_role = 0
    role_checked = 0
    n_fetched = 0
    results = []
    t0 = time.time()
    for i, r in enumerate(sample, 1):
        ent = fetch_revision(r["revision_id"])
        if ent is None:
            results.append({"audit_id": r["audit_id"], "status": "fetch-failed"})
            continue
        n_fetched += 1
        ind_verdict = independent_primary_verdict(ent, r["property_id"],
                                                  r["target_object_id"])
        stored_verdict = ("valid" if r.get("endpoint_primary") == "valid"
                          else "invalid")
        v_ok = ind_verdict == stored_verdict
        agree_verdict += v_ok

        # role check only where the stored class is a role-erasure
        r_ok = None
        if r["operational_class"] in ("role-erased-qualifier",
                                      "role-erased-reference"):
            role_checked += 1
            roles = independent_object_role(ent, r["property_id"],
                                            r["target_object_id"])
            expected = ("qualifier" if r["operational_class"].endswith("qualifier")
                        else "reference")
            r_ok = expected in roles and "main" not in roles
            agree_role += bool(r_ok)

        results.append({"audit_id": r["audit_id"],
                        "stored_class": r["operational_class"],
                        "stored_verdict": stored_verdict,
                        "independent_verdict": ind_verdict,
                        "verdict_agree": v_ok, "role_agree": r_ok})
        if i % 10 == 0:
            print(f"  {i}/{len(sample)}  {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for x in results:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\n=== extraction verification ({n_fetched} fetched) ===")
    print(f"  primary-verdict agreement : {wilson(agree_verdict, n_fetched).pct()}")
    if role_checked:
        print(f"  role-erasure agreement    : {wilson(agree_role, role_checked).pct()}")
    disagree = [x for x in results if x.get("verdict_agree") is False]
    print(f"\ndisagreements: {len(disagree)}")
    for x in disagree[:10]:
        print(f"  {x['audit_id']}  stored={x['stored_verdict']} "
              f"independent={x['independent_verdict']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
