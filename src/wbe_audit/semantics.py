"""Role and rank decomposition of Wikidata entity JSON.

The WikiBigEdit extractor reads a property and entity id out of a 200-char window
around each 'wikibase-entity', which looks identical for main, qualifier and
reference snaks and ignores rank. This module recovers what that misses: the role
of each (property, object) pair, its rank, and the three answer sets we score
against (rho_WBE, rho_all, rho_truthy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

Role = Literal["main", "qualifier", "reference"]
Rank = Literal["preferred", "normal", "deprecated"]

# Datavalue type emitted for every entity-valued snak; the released extractor
# keys on the "wikibase-entity" prefix of this string.
ENTITYID_TYPE = "wikibase-entityid"


@dataclass(frozen=True)
class Occurrence:
    """One entity-valued snak, with the structural context the scanner discards."""

    role: Role
    property_id: str          # property of the snak itself
    value_id: str             # entity id in the snak's datavalue
    entity_type: str          # item / property / lexeme / form / sense
    statement_property: str   # property of the statement the snak hangs off
    statement_guid: str | None
    rank: Rank | None         # rank of the carrying statement
    reference_hash: str | None = None

    @property
    def is_item(self) -> bool:
        return self.entity_type == "item"


def _entity_value(snak: dict[str, Any]) -> tuple[str, str] | None:
    """Return (entity_id, entity_type) for an entity-valued snak, else None."""
    if snak.get("snaktype") != "value":
        return None  # somevalue / novalue carry no datavalue
    dv = snak.get("datavalue")
    if not isinstance(dv, dict) or dv.get("type") != ENTITYID_TYPE:
        return None
    val = dv.get("value")
    if not isinstance(val, dict):
        return None
    eid = val.get("id")
    if not eid:
        # Older serializations carry only numeric-id + entity-type.
        etype = val.get("entity-type")
        num = val.get("numeric-id")
        if etype and num is not None:
            prefix = {"item": "Q", "property": "P", "lexeme": "L"}.get(etype)
            if prefix:
                eid = f"{prefix}{num}"
        if not eid:
            return None
    return eid, val.get("entity-type") or _type_from_id(eid)


def _type_from_id(eid: str) -> str:
    return {"Q": "item", "P": "property", "L": "lexeme"}.get(eid[:1], "unknown")


def iter_claims(entity: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (statement_property, statement) over an entity's claims."""
    claims = entity.get("claims") or entity.get("statements") or {}
    if not isinstance(claims, dict):
        return
    for prop, statements in claims.items():
        if not isinstance(statements, list):
            continue
        for st in statements:
            if isinstance(st, dict):
                yield prop, st


def occurrences(entity: dict[str, Any]) -> list[Occurrence]:
    """Every entity-valued snak in the entity, tagged with role and rank.

    This is the population the released extractor draws from without
    distinguishing its members.
    """
    out: list[Occurrence] = []
    for stmt_prop, st in iter_claims(entity):
        guid = st.get("id")
        rank = st.get("rank")

        main = st.get("mainsnak")
        if isinstance(main, dict):
            ev = _entity_value(main)
            if ev:
                out.append(Occurrence("main", main.get("property") or stmt_prop, ev[0],
                                      ev[1], stmt_prop, guid, rank))

        quals = st.get("qualifiers") or {}
        if isinstance(quals, dict):
            for qprop, qsnaks in quals.items():
                if not isinstance(qsnaks, list):
                    continue
                for qs in qsnaks:
                    if not isinstance(qs, dict):
                        continue
                    ev = _entity_value(qs)
                    if ev:
                        out.append(Occurrence("qualifier", qs.get("property") or qprop,
                                              ev[0], ev[1], stmt_prop, guid, rank))

        refs = st.get("references") or []
        if isinstance(refs, list):
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                rhash = ref.get("hash")
                snaks = ref.get("snaks") or {}
                if not isinstance(snaks, dict):
                    continue
                for rprop, rsnaks in snaks.items():
                    if not isinstance(rsnaks, list):
                        continue
                    for rs in rsnaks:
                        if not isinstance(rs, dict):
                            continue
                        ev = _entity_value(rs)
                        if ev:
                            out.append(Occurrence("reference",
                                                  rs.get("property") or rprop, ev[0],
                                                  ev[1], stmt_prop, guid, rank, rhash))
    return out


# --------------------------------------------------------------------------
# Endpoint answer sets
# --------------------------------------------------------------------------

@dataclass
class AnswerSets:
    """The three endpoint answer sets for one (subject, property)."""

    primary: set[str] = field(default_factory=set)   # non-deprecated main snaks
    all_main: set[str] = field(default_factory=set)  # identical population to primary
    truthy: set[str] = field(default_factory=set)    # best-rank main snaks
    deprecated_only: set[str] = field(default_factory=set)
    has_preferred: bool = False
    n_statements: int = 0


def answer_sets(entity: dict[str, Any], prop: str) -> AnswerSets:
    """Compute rho_WBE / rho_all / rho_truthy answer sets for (entity, prop).

    Only ``main`` role occurrences count: a qualifier or reference value is not
    an assertion that the page subject stands in relation ``prop`` to that value.
    """
    a = AnswerSets()
    non_dep: set[str] = set()
    preferred: set[str] = set()
    normal: set[str] = set()

    for occ in occurrences(entity):
        if occ.role != "main" or occ.statement_property != prop:
            continue
        a.n_statements += 1
        if occ.rank == "deprecated":
            a.deprecated_only.add(occ.value_id)
            continue
        non_dep.add(occ.value_id)
        if occ.rank == "preferred":
            preferred.add(occ.value_id)
        else:
            normal.add(occ.value_id)

    a.primary = set(non_dep)
    a.all_main = set(non_dep)
    a.has_preferred = bool(preferred)
    a.truthy = set(preferred) if preferred else set(normal)
    a.deprecated_only -= non_dep
    return a


def endpoint_primary(a: AnswerSets, obj: str) -> str:
    """rho_WBE: valid only when the non-deprecated main answer set is exactly {obj}."""
    if a.primary == {obj}:
        return "valid"
    if not a.primary and obj in a.deprecated_only:
        # Supported only by a deprecated statement: the benchmark states no rank
        # policy, so this is reported separately rather than silently counted bad.
        return "design-ambiguous"
    return "invalid"


def endpoint_all_main(a: AnswerSets, obj: str) -> bool:
    """rho_all sensitivity: membership among non-deprecated main snaks."""
    return obj in a.all_main


def endpoint_truthy(a: AnswerSets, obj: str) -> bool:
    """rho_truthy sensitivity: membership among best-rank main snaks."""
    return obj in a.truthy


# --------------------------------------------------------------------------
# Role support for a released (property, object) pairing
# --------------------------------------------------------------------------

@dataclass
class RoleSupport:
    """Where a released (prop, obj) pairing actually occurs in the entity.

    The extractor reads the property id and the entity id out of the *same*
    200-character window, so a released pairing whose property and object
    co-occur in a qualifier or reference snak has a concrete non-main origin
    that does not depend on recovering the original text trace.
    """

    roles: set[Role] = field(default_factory=set)
    main_ranks: set[str] = field(default_factory=set)
    qualifier_of: set[str] = field(default_factory=set)   # statement props carrying it
    reference_of: set[str] = field(default_factory=set)
    guids: set[str] = field(default_factory=set)

    @property
    def main_supported(self) -> bool:
        return "main" in self.roles

    @property
    def non_main_only(self) -> bool:
        return bool(self.roles) and not self.main_supported


def role_support(entity: dict[str, Any], prop: str, obj: str) -> RoleSupport:
    """Find every snak in the entity whose property is ``prop`` and value is ``obj``."""
    rs = RoleSupport()
    for occ in occurrences(entity):
        if occ.property_id != prop or occ.value_id != obj:
            continue
        rs.roles.add(occ.role)
        if occ.statement_guid:
            rs.guids.add(occ.statement_guid)
        if occ.role == "main":
            if occ.rank:
                rs.main_ranks.add(occ.rank)
        elif occ.role == "qualifier":
            rs.qualifier_of.add(occ.statement_property)
        else:
            rs.reference_of.add(occ.statement_property)
    return rs
