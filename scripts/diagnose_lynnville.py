"""Why is (Q144370, P39, Q108350406) not reproduced at t1?

Checks the released row verbatim, identifies the object item, and looks for the
pairing at t0, t1 and now -- including in any role, and on any statement.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.replay import scan_text, structural_occurrences  # noqa: E402
from wbe_audit.semantics import occurrences  # noqa: E402
from wbe_audit.wikidata import WikidataClient  # noqa: E402

SUBJ, PROP, OBJ = "Q144370", "P39", "Q108350406"

d = json.load(open("data/raw/wiki_big_edit_20240201_20240220.json", encoding="utf-8"))
row = d[10]
print("released row 10 verbatim:")
print(json.dumps(row, ensure_ascii=False, indent=2))

client = WikidataClient(min_interval=1.2)

# What is the object item?
print("\n--- identify object ---")
snap_obj = client.snapshot(OBJ, "2024-02-20T00:00:00Z")
if snap_obj.ok:
    e = snap_obj.entity
    lab = (e.get("labels", {}).get("en") or {}).get("value")
    desc = (e.get("descriptions", {}).get("en") or {}).get("value")
    print(f"  {OBJ}: label={lab!r} desc={desc!r}")
    inst = [o.value_id for o in occurrences(e)
            if o.role == "main" and o.statement_property == "P31"]
    print(f"  P31 (instance of): {inst}")
else:
    print(f"  could not fetch {OBJ}: {snap_obj.error}")

# Where does the pairing live, across time?
for label, ts in [("t0 2024-02-01", "2024-02-01T00:00:00Z"),
                  ("t1 2024-02-20", "2024-02-20T00:00:00Z"),
                  ("now",           "2026-07-22T00:00:00Z")]:
    print(f"\n--- subject {SUBJ} @ {label} ---")
    snap = client.snapshot(SUBJ, ts, keep_content=True)
    if not snap.ok:
        print(f"  unavailable: {snap.error}")
        continue
    print(f"  revision {snap.revision_id} @ {snap.timestamp}")
    occs = occurrences(snap.entity)
    print(f"  entity-valued snaks: {len(occs)}")

    exact = [o for o in occs if o.property_id == PROP and o.value_id == OBJ]
    print(f"  snaks with property={PROP} and value={OBJ}: {len(exact)}")
    for o in exact:
        print(f"    role={o.role} on statement {o.statement_property} "
              f"rank={o.rank} guid={o.statement_guid}")

    any_prop = [o for o in occs if o.property_id == PROP]
    print(f"  any snak with property={PROP}: {len(any_prop)}")
    for o in any_prop[:8]:
        print(f"    role={o.role} value={o.value_id} on {o.statement_property}")

    any_obj = [o for o in occs if o.value_id == OBJ]
    print(f"  any snak with value={OBJ}: {len(any_obj)}")
    for o in any_obj[:8]:
        print(f"    role={o.role} property={o.property_id} on {o.statement_property}")

    # does the raw scan produce it?
    if snap.content:
        hits = [h for h in scan_text(snap.content) if not h.sentinel]
        got = [(h.occurrence_index, h.relation, h.objective) for h in hits
               if h.emitted and h.relation == PROP]
        print(f"  raw-text scan hits with relation={PROP}: {got[:8]}")
        pairs = {(h.relation, h.objective) for h in hits if h.emitted}
        print(f"  scan reproduces ({PROP},{OBJ}): {(PROP, OBJ) in pairs}")
        struct = structural_occurrences(snap.entity)
        print(f"  alignment: {len(hits)} scan vs {len(struct)} structural "
              f"-> {'OK' if len(hits) == len(struct) else 'MISALIGNED'}")
