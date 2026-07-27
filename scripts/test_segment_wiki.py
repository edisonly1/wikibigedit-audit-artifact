"""Does gensim's Wikipedia markup filter change what the scanner extracts?

`download_wikidata.sh` runs `gensim.scripts.segment_wiki` over the MediaWiki page
dump before `extract_triplets.py` scans it. segment_wiki applies
`gensim.corpora.wikicorpus.filter_wiki`, a *Wikipedia wikitext* cleaner, to text
that is actually a JSON blob. This compares the scanner's output on the raw stored
page text against its output on the filtered text.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gensim.corpora.wikicorpus import filter_wiki  # noqa: E402
from gensim.scripts.segment_wiki import segment  # noqa: E402

from wbe_audit.replay import (emitted_pairs, scan_text,  # noqa: E402
                              structural_occurrences)
from wbe_audit.wikidata import WikidataClient  # noqa: E402

ENTITY = sys.argv[1] if len(sys.argv) > 1 else "Q144370"
TS = sys.argv[2] if len(sys.argv) > 2 else "2024-02-20T00:00:00Z"
TARGET = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("P39", "Q108350406")

client = WikidataClient(min_interval=1.2)
snap = client.snapshot(ENTITY, TS, keep_content=True)
if not snap.ok:
    raise SystemExit(f"fetch failed: {snap.error}")
raw = snap.content
print(f"{ENTITY} @ rev {snap.revision_id} ({snap.timestamp})")
print(f"raw stored page text : {len(raw):,} chars")

filtered = filter_wiki(raw, promote_remaining=True, simplify_links=True)
print(f"after filter_wiki    : {len(filtered):,} chars "
      f"({100*(len(filtered)-len(raw))/len(raw):+.1f}%)")

try:
    titles, texts, _ = segment(raw, include_interlinks=True)
    seg_text = texts[0] if texts else ""
    print(f"segment_wiki section_texts[0]: {len(seg_text):,} chars "
          f"(sections: {len(texts)}, titles: {titles[:3]})")
except Exception as exc:
    seg_text = filtered
    print(f"segment() raised {type(exc).__name__}: {exc}; falling back to filter_wiki")

struct = structural_occurrences(snap.entity)
print(f"structural entity-valued snaks: {len(struct)}")

for name, text in [("RAW", raw), ("filter_wiki", filtered), ("segment_wiki", seg_text)]:
    hits = [h for h in scan_text(text) if not h.sentinel]
    pairs = emitted_pairs(scan_text(text))
    tgt = TARGET in pairs
    print(f"\n--- {name} ---")
    print(f"  marker occurrences : {len(hits)}   (structural: {len(struct)})")
    print(f"  emitted pairs      : {len(pairs)}")
    print(f"  non-emitting hits  : {sum(not h.emitted for h in hits)}")
    print(f"  trim-branch hits   : {sum(h.trimmed for h in hits)}")
    print(f"  contains {TARGET}: {tgt}")
    if tgt:
        for h in hits:
            if h.emitted and (h.relation, h.objective) == TARGET:
                print(f"    at occurrence {h.occurrence_index}, char {h.marker_pos}")
                print(f"    window: ...{text[h.window_start:h.window_end][-180:]!r}")
                break
    rels = {h.relation for h in hits if h.emitted}
    print(f"  distinct relations extracted: {len(rels)}")
    bogus = [r for r in rels if not (r.startswith('P') and r[1:].isdigit())]
    print(f"  malformed relation strings  : {len(bogus)} {sorted(bogus)[:6]}")
    objs = {h.objective for h in hits if h.emitted}
    bogus_o = [o for o in objs
               if not (o[:1] in 'QPL' and o[1:].isdigit())]
    print(f"  malformed object strings    : {len(bogus_o)} {sorted(bogus_o)[:6]}")
