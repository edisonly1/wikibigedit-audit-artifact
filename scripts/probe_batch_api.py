"""Check whether prop=revisions returns one historical revision per page for
multiple titles in a single request. If it does, the full-benchmark screen is
feasible at ~50 entities per API call instead of one."""
import json
import requests

API = "https://www.wikidata.org/w/api.php"
S = requests.Session()
S.headers.update({"User-Agent": "WikiBigEdit-role-aware-audit/0.1 (research)"})

titles = ["Q42", "Q1000254", "Q64", "Q5", "Q1"]
params = {
    "action": "query",
    "prop": "revisions",
    "titles": "|".join(titles),
    "rvstart": "2024-02-20T00:00:00Z",
    "rvdir": "older",
    "rvprop": "ids|timestamp|content",
    "rvslots": "main",
    "format": "json",
    "formatversion": 2,
}
r = S.get(API, params=params, timeout=90)
print("status:", r.status_code, " bytes:", len(r.content))
d = r.json()
if "warnings" in d:
    print("WARNINGS:", json.dumps(d["warnings"])[:500])
if "error" in d:
    print("ERROR:", json.dumps(d["error"])[:500])
pages = d.get("query", {}).get("pages", [])
print("pages returned:", len(pages))
for p in pages:
    revs = p.get("revisions") or []
    print(f"  {p.get('title'):12s} nrevs={len(revs)}", end="")
    if revs:
        rev = revs[0]
        content = (rev.get("slots", {}).get("main", {}) or {}).get("content")
        ok = False
        if content:
            try:
                ent = json.loads(content)
                ok = "claims" in ent
            except json.JSONDecodeError:
                ok = False
        print(f"  revid={rev.get('revid')} ts={rev.get('timestamp')} "
              f"content={'OK' if ok else 'BAD'} len={len(content or '')}")
    else:
        print()
