"""Validate the extraction replay on one documented candidate.

Lynnville (Q144370) is released as holding the position (P39) of mayor
(Q108350406) in interval 2024-02-01..2024-02-20. If the role-erasure account is
correct, the replay should reproduce that pair and align it to a *qualifier*
occurrence rather than a main snak.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.replay import (emitted_pairs, replay, scan_text,  # noqa: E402
                              structural_occurrences)
from wbe_audit.semantics import answer_sets, role_support  # noqa: E402
from wbe_audit.wikidata import WikidataClient  # noqa: E402

ENTITY = sys.argv[1] if len(sys.argv) > 1 else "Q144370"
TS = sys.argv[2] if len(sys.argv) > 2 else "2024-02-20T00:00:00Z"
PROP = sys.argv[3] if len(sys.argv) > 3 else "P39"
OBJ = sys.argv[4] if len(sys.argv) > 4 else "Q108350406"

client = WikidataClient(min_interval=1.0)
snap = client.snapshot(ENTITY, TS, use_cache=False, keep_content=True)
print(f"entity {ENTITY} @ {TS}")
print(f"  revision {snap.revision_id} ts={snap.timestamp} error={snap.error}")
if not snap.ok:
    raise SystemExit("could not fetch entity")

content = snap.content
print(f"  stored page text: {len(content)} chars")

# --- structural view -------------------------------------------------------
a = answer_sets(snap.entity, PROP)
rs = role_support(snap.entity, PROP, OBJ)
print(f"\nSTRUCTURAL")
print(f"  rho_WBE main answer set for {PROP}: {sorted(a.primary) or '(empty)'}")
print(f"  rho_truthy                        : {sorted(a.truthy) or '(empty)'}")
print(f"  main statements for {PROP}        : {a.n_statements}")
print(f"  roles supporting ({PROP},{OBJ})   : {sorted(rs.roles) or '(none)'}")
print(f"  qualifier of statements          : {sorted(rs.qualifier_of)}")
print(f"  reference of statements          : {sorted(rs.reference_of)}")

# --- replay ----------------------------------------------------------------
hits = scan_text(content)
struct = structural_occurrences(snap.entity)
real = [h for h in hits if not h.sentinel]
print(f"\nREPLAY")
print(f"  marker occurrences in page text : {len(real)}")
print(f"  entity-valued snaks in structure: {len(struct)}")
print(f"  aligned                         : {len(real) == len(struct)}")
print(f"  negative-window hits            : {sum(h.negative_window for h in real)}")
print(f"  trim-branch hits                : {sum(h.trimmed for h in real)}")
print(f"  non-emitting hits               : {sum(not h.emitted for h in real)}")

pairs = emitted_pairs(hits)
print(f"  distinct pairs emitted          : {len(pairs)}")
print(f"  ({PROP},{OBJ}) in emitted output : {(PROP, OBJ) in pairs}")

tr = replay(snap.entity, content, PROP, OBJ)
print(f"\nTRACE DIAGNOSIS: {tr.diagnosis}")
print(f"  reproduced      : {tr.reproduced}")
print(f"  hit indices     : {tr.hit_indices}")
print(f"  origin roles    : {tr.origin_roles}")
print(f"  misassociated   : {tr.misassociated}")

for i in tr.hit_indices[:3]:
    h = real[i]
    occ = struct[i] if i < len(struct) else None
    print(f"\n  --- occurrence {i} at char {h.marker_pos} ---")
    print(f"  window    : ...{content[h.window_start:h.window_end][-170:]!r}")
    print(f"  extracted : relation={h.relation!r} objective={h.objective!r}")
    if occ:
        print(f"  structural: role={occ.role} prop={occ.property_id} "
              f"value={occ.value_id} on statement {occ.statement_property} "
              f"rank={occ.rank}")
        print(f"  guid      : {occ.statement_guid}")

# how many of this entity's emitted pairs are NOT main-snak facts?
if len(real) == len(struct):
    by_role = {}
    for h, occ in zip(real, struct):
        if h.emitted:
            by_role[occ.role] = by_role.get(occ.role, 0) + 1
    print(f"\n  emitted occurrences by structural role: {by_role}")
