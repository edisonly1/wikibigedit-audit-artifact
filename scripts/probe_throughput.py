"""Find a historical-fetch configuration that Wikidata does not throttle.

Compares api.php prop=revisions content fetches against the index.php action=raw
endpoint (CDN-fronted, usually far cheaper), and serial vs concurrent access.
"""
from __future__ import annotations

import time

import requests

UA_GOOD = ("WikiBigEdit-audit/0.1 (https://github.com/; research on knowledge-editing "
           "benchmark provenance; contact: researcher@example.org) python-requests")

QIDS = ["Q144370", "Q16870617", "Q2478944", "Q1101850", "Q7575962",
        "Q16870314", "Q1913499", "Q2323309", "Q1927469", "Q5603833"]
TS = "2024-02-20T00:00:00Z"


def make_session(ua: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip"})
    return s


def timed(fn, label: str, n: int):
    t0 = time.time()
    ok = 0
    codes = {}
    for i in range(n):
        code = fn(i)
        codes[code] = codes.get(code, 0) + 1
        if code == 200:
            ok += 1
    el = time.time() - t0
    print(f"  {label:52s} {ok}/{n} ok  {el:6.1f}s  "
          f"{n/el:5.2f} req/s  {codes}")
    return el


print("=== api.php prop=revisions with content (what the audit uses) ===")
for ua_label, ua in [("terse UA", "WikiBigEdit-role-aware-audit/0.1 (research)"),
                     ("descriptive UA", UA_GOOD)]:
    s = make_session(ua)

    def f(i, s=s):
        r = s.get("https://www.wikidata.org/w/api.php", params={
            "action": "query", "prop": "revisions", "titles": QIDS[i % len(QIDS)],
            "rvstart": TS, "rvdir": "older", "rvlimit": 1,
            "rvprop": "ids|timestamp|content", "rvslots": "main",
            "format": "json", "formatversion": 2}, timeout=60)
        return r.status_code

    timed(f, f"api.php content, serial, no pause [{ua_label}]", 10)

print("\n=== api.php prop=revisions, ids+timestamp only (no content) ===")
s = make_session(UA_GOOD)


def f_ids(i):
    r = s.get("https://www.wikidata.org/w/api.php", params={
        "action": "query", "prop": "revisions", "titles": QIDS[i % len(QIDS)],
        "rvstart": TS, "rvdir": "older", "rvlimit": 1,
        "rvprop": "ids|timestamp", "format": "json", "formatversion": 2},
        timeout=60)
    return r.status_code


timed(f_ids, "api.php ids-only, serial", 10)

print("\n=== index.php action=raw by oldid (CDN-fronted) ===")
# resolve revids once
revids = []
for q in QIDS:
    r = s.get("https://www.wikidata.org/w/api.php", params={
        "action": "query", "prop": "revisions", "titles": q,
        "rvstart": TS, "rvdir": "older", "rvlimit": 1,
        "rvprop": "ids", "format": "json", "formatversion": 2}, timeout=60)
    try:
        revids.append(r.json()["query"]["pages"][0]["revisions"][0]["revid"])
    except Exception:
        revids.append(None)
    time.sleep(0.3)
revids = [x for x in revids if x]
print(f"  resolved {len(revids)} revids")


def f_raw(i):
    rid = revids[i % len(revids)]
    r = s.get("https://www.wikidata.org/w/index.php",
              params={"oldid": rid, "action": "raw"}, timeout=60)
    return r.status_code


timed(f_raw, "index.php action=raw, serial", 20)

print("\n=== Special:EntityData with revision (CDN-fronted) ===")


def f_ed(i):
    rid = revids[i % len(revids)]
    q = QIDS[i % len(revids)]
    r = s.get(f"https://www.wikidata.org/wiki/Special:EntityData/{q}.json",
              params={"revision": rid}, timeout=60)
    return r.status_code


timed(f_ed, "Special:EntityData?revision, serial", 20)
