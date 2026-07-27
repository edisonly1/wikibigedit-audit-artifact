"""Unit tests for the endpoint semantics and the scanner replay.

These fix the behaviour that the prevalence numbers depend on, so the classifier
can be frozen before the blinded validation phase.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.replay import (emitted_pairs, replay, scan_text,
                              structural_occurrences)
from wbe_audit.semantics import (answer_sets, endpoint_all_main,
                                 endpoint_primary, endpoint_truthy,
                                 occurrences, role_support)


def snak(prop: str, qid: str) -> dict:
    return {
        "snaktype": "value",
        "property": prop,
        "hash": "0" * 40,
        "datavalue": {
            "value": {"entity-type": "item",
                      "numeric-id": int(qid[1:]), "id": qid},
            "type": "wikibase-entityid",
        },
        "datatype": "wikibase-item",
    }


def statement(prop: str, qid: str, rank: str = "normal", guid: str = "G1",
              qualifiers: dict | None = None,
              references: list | None = None) -> dict:
    st = {"mainsnak": snak(prop, qid), "type": "statement"}
    if qualifiers:
        st["qualifiers"] = qualifiers
    st["id"] = guid
    st["rank"] = rank
    if references:
        st["references"] = references
    return st


def entity(claims: dict) -> dict:
    return {"type": "item", "id": "Q1", "labels": {}, "claims": claims}


# --------------------------------------------------------------------------

def test_unique_main_is_primary_valid():
    e = entity({"P1": [statement("P1", "Q10")]})
    a = answer_sets(e, "P1")
    assert a.primary == {"Q10"}
    assert endpoint_primary(a, "Q10") == "valid"
    assert endpoint_all_main(a, "Q10")
    assert endpoint_truthy(a, "Q10")


def test_two_main_values_fail_rho_wbe_but_pass_rho_all():
    e = entity({"P1": [statement("P1", "Q10", guid="G1"),
                       statement("P1", "Q11", guid="G2")]})
    a = answer_sets(e, "P1")
    assert a.primary == {"Q10", "Q11"}
    assert endpoint_primary(a, "Q10") == "invalid"   # not unique
    assert endpoint_all_main(a, "Q10")               # but is a main value
    assert endpoint_truthy(a, "Q10")


def test_deprecated_excluded_from_primary():
    e = entity({"P1": [statement("P1", "Q10", rank="deprecated")]})
    a = answer_sets(e, "P1")
    assert a.primary == set()
    assert a.deprecated_only == {"Q10"}
    assert endpoint_primary(a, "Q10") == "design-ambiguous"
    assert not endpoint_all_main(a, "Q10")


def test_truthy_prefers_preferred_rank():
    e = entity({"P1": [statement("P1", "Q10", rank="preferred", guid="G1"),
                       statement("P1", "Q11", rank="normal", guid="G2")]})
    a = answer_sets(e, "P1")
    assert a.primary == {"Q10", "Q11"}
    assert a.truthy == {"Q10"}
    assert endpoint_truthy(a, "Q10")
    assert not endpoint_truthy(a, "Q11")   # normal is shadowed by preferred


def test_qualifier_value_is_not_a_main_fact():
    """The core role-erasure case: P39/Q30185 exists only as a qualifier."""
    e = entity({"P6": [statement("P6", "Q99", guid="G1",
                                 qualifiers={"P39": [snak("P39", "Q30185")]})]})
    a = answer_sets(e, "P39")
    assert a.primary == set()             # no P39 main statement
    assert endpoint_primary(a, "Q30185") == "invalid"

    rs = role_support(e, "P39", "Q30185")
    assert rs.roles == {"qualifier"}
    assert rs.non_main_only
    assert rs.qualifier_of == {"P6"}      # qualifier hangs off a P6 statement


def test_reference_value_is_not_a_main_fact():
    ref = [{"hash": "h1", "snaks": {"P248": [snak("P248", "Q555")]}}]
    e = entity({"P1": [statement("P1", "Q10", guid="G1", references=ref)]})
    rs = role_support(e, "P248", "Q555")
    assert rs.roles == {"reference"}
    assert rs.reference_of == {"P1"}
    a = answer_sets(e, "P248")
    assert endpoint_primary(a, "Q555") == "invalid"


def test_occurrence_counts_cover_all_three_roles():
    ref = [{"hash": "h1", "snaks": {"P248": [snak("P248", "Q555")]}}]
    e = entity({"P6": [statement("P6", "Q99", guid="G1",
                                 qualifiers={"P39": [snak("P39", "Q30185")]},
                                 references=ref)]})
    occs = occurrences(e)
    assert {o.role for o in occs} == {"main", "qualifier", "reference"}
    assert len(occs) == 3


# --------------------------------------------------------------------------
# scanner replay

def test_scanner_extracts_qualifier_pair_identically_to_main_pair():
    """A qualifier snak is indistinguishable from a main snak to the scanner."""
    ref = [{"hash": "h1", "snaks": {"P248": [snak("P248", "Q555")]}}]
    e = entity({"P6": [statement("P6", "Q99", guid="G1",
                                 qualifiers={"P39": [snak("P39", "Q30185")]},
                                 references=ref)]})
    text = json.dumps(e, separators=(",", ":"))
    pairs = emitted_pairs(scan_text(text))
    # the trailing truncation drops the last emitted pair, so check membership of
    # the earlier ones rather than exact set equality
    assert ("P6", "Q99") in pairs
    assert ("P39", "Q30185") in pairs


def test_replay_aligns_and_diagnoses_qualifier_origin():
    ref = [{"hash": "h1", "snaks": {"P248": [snak("P248", "Q555")]}}]
    e = entity({"P6": [statement("P6", "Q99", guid="G1",
                                 qualifiers={"P39": [snak("P39", "Q30185")]},
                                 references=ref)],
                "P31": [statement("P31", "Q515", guid="G2")]})
    text = json.dumps(e, separators=(",", ":"))
    hits = [h for h in scan_text(text) if not h.sentinel]
    struct = structural_occurrences(e)
    assert len(hits) == len(struct), "marker/structure alignment must hold"

    tr = replay(e, text, "P39", "Q30185")
    assert tr.reproduced
    assert tr.aligned
    assert tr.diagnosis == "qualifier-occurrence"
    assert tr.origin_roles == ["qualifier"]


def test_replay_reports_not_reproduced_for_absent_pair():
    e = entity({"P31": [statement("P31", "Q515", guid="G1")]})
    text = json.dumps(e, separators=(",", ":"))
    tr = replay(e, text, "P39", "Q30185")
    assert not tr.reproduced
    assert tr.diagnosis == "not-reproduced"


def test_somevalue_and_novalue_snaks_are_ignored():
    e = entity({"P1": [{"mainsnak": {"snaktype": "somevalue", "property": "P1",
                                     "datatype": "wikibase-item"},
                        "type": "statement", "id": "G1", "rank": "normal"}]})
    assert answer_sets(e, "P1").primary == set()
    assert occurrences(e) == []


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
