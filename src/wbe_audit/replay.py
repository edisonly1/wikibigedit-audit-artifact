"""Replay the released extractor over exact revision text.

We don't need the pipeline's unreleased intermediate files. The MediaWiki API
returns the exact stored page text for a revision, so we re-run the 200-char
scanner on it and check whether a released row is reproduced.

`scan_text` ports the scanner (recording offsets and windows) and
`structural_occurrences` walks the parsed entity in document order. Since
`json.loads` keeps key order, the n-th 'wikibase-entityid' in the text is the n-th
snak in the structure, so the two line up index-for-index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from .semantics import ENTITYID_TYPE, Occurrence, _entity_value, _type_from_id

MARKER = "wikibase-entity"
WINDOW = 200


@dataclass
class ScanHit:
    """One occurrence as the released scanner sees it."""

    occurrence_index: int
    marker_pos: int
    window_start: int
    window_end: int
    relation: str
    objective: str
    emitted: bool           # did the scanner append this pair?
    trimmed: bool           # did the `"},"type":` compensation branch fire?
    negative_window: bool   # marker_pos < WINDOW, so the slice wrapped
    sentinel: bool          # this is the index == -1 tail parse


def scan_text(texts: str) -> list[ScanHit]:
    """Port of the released scanner, preserving its exact index arithmetic.

    `extract_triplets.py` computes `target = texts[index-200:index]` and parses
    four `find` results out of it. The final iteration runs with `index == -1`,
    parses the tail of the document, and is only afterwards removed by the
    unconditional `small_list[:-1]`.
    """
    hits: list[ScanHit] = []
    index = -1
    n = 0
    while True:
        index = texts.find(MARKER, index + 1)
        target = texts[index - WINDOW:index]

        idx1 = target.find('"property"')
        idx2 = target.find("hash")
        idx3 = target.find('"id":"')
        idx4 = target.find(',"type":')
        relation = target[idx1 + 12: idx2 - 3]
        objective = target[idx3 + 6: idx4 - 2]

        emitted = False
        trimmed = False
        if relation == "" or objective == "":
            pass
        elif '"},"type":' in objective:
            objective = objective[:-10]
            emitted, trimmed = True, True
        else:
            emitted = True

        hits.append(ScanHit(
            occurrence_index=n,
            marker_pos=index,
            window_start=index - WINDOW,
            window_end=index,
            relation=relation,
            objective=objective,
            emitted=emitted,
            trimmed=trimmed,
            negative_window=(0 <= index < WINDOW),
            sentinel=(index == -1),
        ))
        n += 1
        if index == -1:
            break
    return hits


def emitted_pairs(hits: list[ScanHit]) -> list[tuple[str, str]]:
    """Reproduce the scanner's per-entity output list, including its quirks.

    Mirrors `small_list = small_list[:-1]` followed by order-preserving dedup.
    """
    pairs = [(h.relation, h.objective) for h in hits if h.emitted]
    pairs = pairs[:-1]  # unconditional truncation in the released code
    seen: list[tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.append(p)
    return seen


def structural_occurrences(entity: dict[str, Any]) -> list[Occurrence]:
    """Entity-valued snaks in *serialization order*.

    Traverses each statement following the actual key order of the parsed JSON
    rather than an assumed mainsnak/qualifiers/references ordering, so the result
    aligns positionally with `scan_text` output.
    """
    out: list[Occurrence] = []
    claims = entity.get("claims") or entity.get("statements") or {}
    if not isinstance(claims, dict):
        return out

    for stmt_prop, statements in claims.items():
        if not isinstance(statements, list):
            continue
        for st in statements:
            if not isinstance(st, dict):
                continue
            guid = st.get("id")
            rank = st.get("rank")
            for key, val in st.items():   # actual serialization order
                if key == "mainsnak" and isinstance(val, dict):
                    ev = _entity_value(val)
                    if ev:
                        out.append(Occurrence("main", val.get("property") or stmt_prop,
                                              ev[0], ev[1], stmt_prop, guid, rank))
                elif key == "qualifiers" and isinstance(val, dict):
                    for qprop, qsnaks in val.items():
                        for qs in (qsnaks or []):
                            if not isinstance(qs, dict):
                                continue
                            ev = _entity_value(qs)
                            if ev:
                                out.append(Occurrence(
                                    "qualifier", qs.get("property") or qprop,
                                    ev[0], ev[1], stmt_prop, guid, rank))
                elif key == "references" and isinstance(val, list):
                    for ref in val:
                        if not isinstance(ref, dict):
                            continue
                        rhash = ref.get("hash")
                        for rprop, rsnaks in (ref.get("snaks") or {}).items():
                            for rs in (rsnaks or []):
                                if not isinstance(rs, dict):
                                    continue
                                ev = _entity_value(rs)
                                if ev:
                                    out.append(Occurrence(
                                        "reference", rs.get("property") or rprop,
                                        ev[0], ev[1], stmt_prop, guid, rank, rhash))
    return out


@dataclass
class TraceResult:
    """Outcome of replaying the scanner for one released (property, object) row."""

    reproduced: bool                    # scanner emits the released pair
    n_marker_occurrences: int
    n_structural_occurrences: int
    aligned: bool                       # counts match, so index join is sound
    hit_indices: list[int]              # occurrences that produced the pair
    origin_roles: list[str]             # structural role at each of those indices
    misassociated: list[bool]           # property and object came from different snaks
    diagnosis: str


def replay(entity: dict[str, Any], content: str, prop: str, obj: str) -> TraceResult:
    """Replay the scanner and diagnose where a released pairing came from."""
    hits = scan_text(content)
    real_hits = [h for h in hits if not h.sentinel]
    structural = structural_occurrences(entity)
    aligned = len(real_hits) == len(structural)

    idxs, roles, misassoc = [], [], []
    for h in real_hits:
        if not h.emitted or h.relation != prop or h.objective != obj:
            continue
        idxs.append(h.occurrence_index)
        if aligned and h.occurrence_index < len(structural):
            occ = structural[h.occurrence_index]
            roles.append(occ.role)
            # the scanner read property and object out of one window; if the snak
            # actually sitting at this position disagrees, the window straddled a
            # boundary
            misassoc.append(not (occ.property_id == prop and occ.value_id == obj))
        else:
            roles.append("unaligned")
            misassoc.append(False)

    reproduced = bool(idxs)
    if not reproduced:
        diagnosis = "not-reproduced"
    elif any(misassoc):
        diagnosis = "window-misassociation"
    elif all(r == "main" for r in roles):
        diagnosis = "mainsnak-occurrence"
    elif any(r == "qualifier" for r in roles):
        diagnosis = "qualifier-occurrence"
    elif any(r == "reference" for r in roles):
        diagnosis = "reference-occurrence"
    else:
        diagnosis = "unresolved"

    return TraceResult(
        reproduced=reproduced,
        n_marker_occurrences=len(real_hits),
        n_structural_occurrences=len(structural),
        aligned=aligned,
        hit_indices=idxs,
        origin_roles=roles,
        misassociated=misassoc,
        diagnosis=diagnosis,
    )
