"""Fetch Wikidata entity JSON at a given revision, with an on-disk cache.

We need the structured entity state at each snapshot date, which the released
pipeline never looks at (it only scans page text). Fetching the revision that was
current at the snapshot boundary recovers that state with roles, ranks and GUIDs
intact.
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

API = "https://www.wikidata.org/w/api.php"

# Wikimedia's robot policy asks for a descriptive User-Agent with a contact. Set
# WBE_CONTACT before any large run.
#
# Measured rate limit (see scripts/probe_sustained.py): a burst of ~10 requests
# succeeds, then api.php returns 429 until the bucket refills at roughly
# 10 requests/minute. This holds for prop=revisions with content, for ids-only
# queries, and for the Special:EntityData path (whose revid lookup still goes
# through api.php) -- so it is a per-endpoint-family budget, not a payload or
# concurrency effect, and there is no faster route for uncached historical state.
#
# Consequences the audit is built around:
#   * historical fetches cannot be batched (rvstart is rejected for multi-title
#     queries), so one (entity, snapshot) costs one request: ~600/hour;
#   * current-state lookups CAN be batched 50/request via wbgetentities, which is
#     why the discovery tier uses them and reaches ~500 entities/minute;
#   * default spacing below sits just under the refill rate so runs are
#     predictable instead of burning retries on 429 backoff.
_CONTACT = os.environ.get("WBE_CONTACT", "contact not configured")
UA = (f"WikiBigEdit-role-aware-audit/0.1 "
      f"(benchmark provenance research; {_CONTACT}) python-requests")

CACHE_DIR = os.environ.get("WBE_CACHE", os.path.join("data", "cache"))


@dataclass
class Snapshot:
    """An entity's structured state at the revision current as of ``timestamp``."""

    entity_id: str
    revision_id: int | None
    timestamp: str | None
    entity: dict[str, Any] | None  # parsed entity JSON, None if unavailable
    error: str | None = None
    content: str | None = None     # exact stored page text, for parser replay

    @property
    def ok(self) -> bool:
        return self.entity is not None


def _cache_path(entity_id: str, iso_ts: str) -> str:
    # shard by the numeric block used by the original pipeline (1M entities)
    try:
        block = int(entity_id[1:]) // 100_000
    except ValueError:
        block = 0
    d = os.path.join(CACHE_DIR, iso_ts[:10].replace("-", ""), str(block))
    return os.path.join(d, f"{entity_id}.json.gz")


def _read_cache(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


class _RateLimiter:
    """Global minimum spacing between requests, shared across worker threads.

    Historical revision content cannot be batched (rvstart is rejected for
    multi-title queries), so every (entity, snapshot) costs one request and the
    only lever is to stay under the anonymous rate ceiling rather than trip 429s.
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_at = max(now, self._next_at) + self.min_interval

    def penalise(self, seconds: float) -> None:
        """Push the whole fleet back after a 429."""
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + seconds)


class WikidataClient:
    """Fetches entity JSON as of a timestamp, with disk cache and polite retries."""

    def __init__(self, session: requests.Session | None = None, max_retries: int = 6,
                 sleep_base: float = 1.0, min_interval: float = 6.2):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
        self.max_retries = max_retries
        self.sleep_base = sleep_base
        self.limiter = _RateLimiter(min_interval)
        self.n_requests = 0
        self.n_cache_hits = 0
        self.n_429 = 0

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        last = None
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            try:
                self.n_requests += 1
                r = self.session.get(API, params=params, timeout=90)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    self.n_429 += 1
                    retry_after = r.headers.get("Retry-After")
                    delay = float(retry_after) if (retry_after or "").isdigit() \
                        else self.sleep_base * (2 ** attempt)
                    self.limiter.penalise(min(delay, 120.0))
                    last = "HTTP 429"
                    continue
                last = f"HTTP {r.status_code}"
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(self.sleep_base * (2 ** attempt))
        raise RuntimeError(f"wikidata API failed after {self.max_retries} tries: {last}")

    def snapshot(self, entity_id: str, iso_ts: str, use_cache: bool = True,
                 keep_content: bool = False) -> Snapshot:
        """Entity state at the last revision at or before ``iso_ts``.

        ``iso_ts`` is an ISO-8601 timestamp, e.g. ``2024-02-01T00:00:00Z``. The
        snapshot boundary is interpreted as *inclusive of everything before it*,
        matching a dump generated at that date.
        """
        path = _cache_path(entity_id, iso_ts)
        if use_cache:
            cached = _read_cache(path)
            if cached is not None:
                self.n_cache_hits += 1
                return Snapshot(entity_id, cached.get("revid"), cached.get("timestamp"),
                                cached.get("entity"), cached.get("error"),
                                cached.get("content"))

        params = {
            "action": "query",
            "prop": "revisions",
            "titles": entity_id,
            "rvstart": iso_ts,
            "rvdir": "older",
            "rvlimit": 1,
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "format": "json",
            "formatversion": 2,
        }
        data = self._get(params)
        pages = data.get("query", {}).get("pages", [])
        payload: dict[str, Any] = {"revid": None, "timestamp": None,
                                   "entity": None, "error": None, "content": None}

        if not pages:
            payload["error"] = "no-page"
        else:
            page = pages[0]
            if page.get("missing"):
                payload["error"] = "missing-page"
            else:
                revs = page.get("revisions") or []
                if not revs:
                    payload["error"] = "no-revision-before-timestamp"
                else:
                    rev = revs[0]
                    payload["revid"] = rev.get("revid")
                    payload["timestamp"] = rev.get("timestamp")
                    content = (rev.get("slots", {}).get("main", {}) or {}).get("content")
                    if content is None:
                        payload["error"] = "no-content"
                    else:
                        try:
                            # object_pairs_hook is unnecessary: json.loads already
                            # preserves key order, which the replay relies on to
                            # align structural snaks with serialized occurrences.
                            payload["entity"] = json.loads(content)
                            if keep_content:
                                payload["content"] = content
                        except json.JSONDecodeError as exc:
                            payload["error"] = f"content-not-json: {exc}"

        _write_cache(path, payload)
        return Snapshot(entity_id, payload["revid"], payload["timestamp"],
                        payload["entity"], payload["error"], payload["content"])

    def revisions_between(self, entity_id: str, start_ts: str, end_ts: str,
                          limit: int = 500) -> list[dict[str, Any]]:
        """All revisions of ``entity_id`` in (start_ts, end_ts], oldest first.

        Used by the differencer to classify a change as an atomic insertion /
        replacement versus a composite or revert cycle.
        """
        out: list[dict[str, Any]] = []
        cont: dict[str, Any] = {}
        while True:
            params = {
                "action": "query",
                "prop": "revisions",
                "titles": entity_id,
                "rvstart": start_ts,
                "rvend": end_ts,
                "rvdir": "newer",
                "rvlimit": min(limit, 500),
                "rvprop": "ids|timestamp|comment|user|flags|tags|size",
                "format": "json",
                "formatversion": 2,
                **cont,
            }
            data = self._get(params)
            pages = data.get("query", {}).get("pages", [])
            if pages and not pages[0].get("missing"):
                out.extend(pages[0].get("revisions") or [])
            if "continue" in data and len(out) < limit:
                cont = data["continue"]
            else:
                break
        return out[:limit]
