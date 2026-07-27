"""Classify each released row against the endpoint semantics.

A row is (subject_id, relation_id, object_id) from an interval file. We check
whether Wikidata at t1 actually supports it as a main fact, and if not, why not.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Any

from .semantics import (AnswerSets, answer_sets, endpoint_all_main,
                        endpoint_primary, endpoint_truthy, occurrences,
                        role_support)
from .wikidata import Snapshot, WikidataClient

INTERVAL_RE = re.compile(r"wiki_big_edit_(\d{8})_(\d{8})\.json$")


def interval_bounds(filename: str) -> tuple[str, str]:
    """('2024-02-01T00:00:00Z', '2024-02-20T00:00:00Z') from the interval filename."""
    m = INTERVAL_RE.search(os.path.basename(filename))
    if not m:
        raise ValueError(f"not an interval file: {filename}")
    def iso(d: str) -> str:
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}T00:00:00Z"
    return iso(m.group(1)), iso(m.group(2))


# Operational classes, refined to separate a *role* failure from a
# *multiplicity* failure -- both are rho_WBE-invalid but they are different defects
# and reviewers will read a merged number as inflated.
CLASS_ENDPOINT_VALID = "endpoint-valid"
CLASS_MAIN_NONUNIQUE = "main-supported-nonunique"
CLASS_ROLE_ERASED_QUAL = "role-erased-qualifier"
CLASS_ROLE_ERASED_REF = "role-erased-reference"
CLASS_DEPRECATED_ONLY = "deprecated-only"
CLASS_SUBJECT_MISSING = "subject-unavailable"

# The two failure signatures that must not be pooled. A row whose subject carries
# the released property but not the released object is a *value* failure. A row
# whose subject has no snak with that property at all, in any role, is the
# signature of a wrong-entity attachment: the QA pipeline dropped QIDs after
# extract_triplets and carried identity as label strings only.
CLASS_OBJECT_ABSENT = "property-present-object-absent"
CLASS_PROPERTY_ABSENT = "property-absent-on-subject"


@dataclass
class RowAudit:
    """Verdict plus the structured evidence that produced it."""

    audit_id: str
    source_interval: tuple[str, str]
    source_row_index: int
    subject_id: str
    property_id: str
    target_object_id: str
    tag: str | None

    revision_id: int | None = None
    revision_timestamp: str | None = None

    endpoint_primary_objects: list[str] = None
    endpoint_truthy_objects: list[str] = None
    deprecated_objects: list[str] = None
    n_main_statements: int = 0

    endpoint_primary: str = "unresolved"     # valid | invalid | design-ambiguous
    valid_all_main: bool = False
    valid_truthy: bool = False

    support_roles: list[str] = None
    qualifier_of: list[str] = None
    reference_of: list[str] = None
    main_ranks: list[str] = None
    statement_guids: list[str] = None

    # identity-integrity signals
    property_present_any_role: bool = False
    property_present_main: bool = False
    object_present_any_property: bool = False
    n_entity_snaks: int = 0

    operational_class: str = "unresolved"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_interval"] = list(self.source_interval)
        return d


def classify(row: dict[str, Any], row_index: int, interval_file: str,
             snap: Snapshot) -> RowAudit:
    """Classify one released row against the entity snapshot at t1."""
    t0, t1 = interval_bounds(interval_file)
    s = row.get("subject_id")
    p = row.get("relation_id")
    o = row.get("object_id")
    tag = row.get("tag")
    if not isinstance(tag, str):
        tag = None

    ra = RowAudit(
        audit_id=f"{os.path.basename(interval_file)}:{row_index}:{s}:{p}:{o}",
        source_interval=(t0, t1),
        source_row_index=row_index,
        subject_id=s, property_id=p, target_object_id=o, tag=tag,
        revision_id=snap.revision_id, revision_timestamp=snap.timestamp,
        endpoint_primary_objects=[], endpoint_truthy_objects=[],
        deprecated_objects=[], support_roles=[], qualifier_of=[],
        reference_of=[], main_ranks=[], statement_guids=[],
    )

    if not snap.ok:
        ra.error = snap.error or "no-entity"
        ra.operational_class = CLASS_SUBJECT_MISSING
        return ra

    a: AnswerSets = answer_sets(snap.entity, p)
    rs = role_support(snap.entity, p, o)

    ra.endpoint_primary_objects = sorted(a.primary)
    ra.endpoint_truthy_objects = sorted(a.truthy)
    ra.deprecated_objects = sorted(a.deprecated_only)
    ra.n_main_statements = a.n_statements

    ra.endpoint_primary = endpoint_primary(a, o)
    ra.valid_all_main = endpoint_all_main(a, o)
    ra.valid_truthy = endpoint_truthy(a, o)

    ra.support_roles = sorted(rs.roles)
    ra.qualifier_of = sorted(rs.qualifier_of)
    ra.reference_of = sorted(rs.reference_of)
    ra.main_ranks = sorted(rs.main_ranks)
    ra.statement_guids = sorted(rs.guids)

    occs = occurrences(snap_entity := snap.entity)
    ra.n_entity_snaks = len(occs)
    ra.property_present_any_role = any(x.property_id == p for x in occs)
    ra.property_present_main = any(x.property_id == p and x.role == "main"
                                   for x in occs)
    ra.object_present_any_property = any(x.value_id == o for x in occs)

    ra.operational_class = _operational_class(ra, a, rs, o)
    return ra


def _operational_class(ra: RowAudit, a: AnswerSets, rs, o: str) -> str:
    if ra.endpoint_primary == "valid":
        return CLASS_ENDPOINT_VALID
    # main-supported but the endpoint is not unique: a multiplicity failure under
    # rho_WBE, still a legitimate main fact under rho_all.
    if o in a.all_main:
        return CLASS_MAIN_NONUNIQUE
    if o in a.deprecated_only:
        return CLASS_DEPRECATED_ONLY
    # No main support at all. Did the released (property, object) pairing exist in
    # a non-main role? Property and object are read from the same snak window, so
    # co-occurrence in a qualifier/reference snak is a concrete non-main origin.
    if "qualifier" in rs.roles:
        return CLASS_ROLE_ERASED_QUAL
    if "reference" in rs.roles:
        return CLASS_ROLE_ERASED_REF
    # Neither. Separate "this subject has the property but not this value" from
    # "this subject has no such property at all" -- only the latter indicts the
    # subject's identity rather than the value.
    if ra.property_present_any_role:
        return CLASS_OBJECT_ABSENT
    return CLASS_PROPERTY_ABSENT


def audit_row(client: WikidataClient, row: dict[str, Any], row_index: int,
              interval_file: str) -> RowAudit:
    _, t1 = interval_bounds(interval_file)
    s = row.get("subject_id")
    if not s:
        snap = Snapshot("", None, None, None, "no-subject-id")
    else:
        snap = client.snapshot(s, t1)
    return classify(row, row_index, interval_file, snap)
