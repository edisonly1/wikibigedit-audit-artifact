"""Sustained-throughput comparison for historical entity fetches.

The burst benchmark was misleading: api.php sustains ~9 req/s for ten requests and
then collapses. This measures 60 consecutive requests on each path to find which
one actually sustains.

Path A: api.php prop=revisions with content (one request per entity-snapshot).
Path B: api.php prop=revisions ids-only, then Special:EntityData/{Q}.json?revision=
        (two requests, but the second is CDN-fronted and immutable per revision).
"""
from __future__ import annotations

import json
import random
import time

import requests

UA = "WikiBigEdit-audit/0.1 (benchmark provenance research; akbc2026) python-requests"
S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})

TS = "2024-04-01T00:00:00Z"
rng = random.Random(7)
# spread across the QID space so nothing is warm in a local cache
QIDS = [f"Q{rng.randrange(1000, 900000)}" for _ in range(140)]

N = 60


def path_a(qid: str):
    r = S.get("https://www.wikidata.org/w/api.php", params={
        "action": "query", "prop": "revisions", "titles": qid,
        "rvstart": TS, "rvdir": "older", "rvlimit": 1,
        "rvprop": "ids|timestamp|content", "rvslots": "main",
        "format": "json", "formatversion": 2}, timeout=90)
    return r.status_code, len(r.content)


def path_b(qid: str):
    r1 = S.get("https://www.wikidata.org/w/api.php", params={
        "action": "query", "prop": "revisions", "titles": qid,
        "rvstart": TS, "rvdir": "older", "rvlimit": 1,
        "rvprop": "ids", "format": "json", "formatversion": 2}, timeout=90)
    if r1.status_code != 200:
        return r1.status_code, 0
    try:
        rid = r1.json()["query"]["pages"][0]["revisions"][0]["revid"]
    except Exception:
        return 204, len(r1.content)   # no revision before ts; still a completed unit
    r2 = S.get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
               params={"revision": rid}, timeout=90)
    return r2.status_code, len(r1.content) + len(r2.content)


def run(name, fn, qids):
    codes = {}
    total_bytes = 0
    t0 = time.time()
    for q in qids:
        try:
            code, nb = fn(q)
        except requests.RequestException as exc:
            code, nb = type(exc).__name__, 0
        codes[code] = codes.get(code, 0) + 1
        total_bytes += nb
    el = time.time() - t0
    ok = codes.get(200, 0)
    print(f"\n{name}")
    print(f"  {len(qids)} units in {el:6.1f}s -> {len(qids)/el:5.2f} entities/s")
    print(f"  ok={ok}  codes={codes}")
    print(f"  {total_bytes/1e6:.1f} MB  ({total_bytes/max(el,1)/1e6:.2f} MB/s)")
    return len(qids) / el


print(f"sustained test, {N} entities per path, serial, no artificial pause")
a = run("PATH A: api.php prop=revisions with content", path_a, QIDS[:N])
time.sleep(5)
b = run("PATH B: api.php ids-only + Special:EntityData?revision", path_b, QIDS[N:2*N])

print(f"\nwinner: {'B' if b > a else 'A'}  ({max(a,b)/max(min(a,b),1e-9):.1f}x)")
