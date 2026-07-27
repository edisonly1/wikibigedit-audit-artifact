"""Build the matched probe strata for the reconstructed locality experiment.

WikiBigEdit picks a locality probe by taking an unchanged triple with the same
relation and the fuzzy-nearest subject label (the same-subject branch is nearly
unreachable after the ambiguity filter). To isolate that choice, we build three
strata that share the relation, template and answer format and differ only in the
probe subject: fuzzy (label-nearest), random, and same (the edited subject under a
different property). The contrast of interest is fuzzy vs random.

Probe facts are checked against current Wikidata (batched wbgetentities, 50 per
request); a probe only needs to be a real fact, so historical verification isn't
needed here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.semantics import answer_sets, occurrences  # noqa: E402

API = "https://www.wikidata.org/w/api.php"
S = requests.Session()
S.headers.update({"User-Agent": "WikiBigEdit-audit/0.1 (akbc2026)",
                  "Accept-Encoding": "gzip"})


def get(params, tries=6):
    for a in range(tries):
        try:
            r = S.get(API, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if (ra or "").isdigit() else min(2 ** a, 60))
                continue
        except requests.RequestException:
            pass
        time.sleep(min(2 ** a, 30))
    return {}


def fetch(qids, props, pause=6.5):
    out = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        d = get({"action": "wbgetentities", "ids": "|".join(chunk),
                 "props": props, "languages": "en", "format": "json",
                 "formatversion": 2})
        ents = d.get("entities") or {}
        if not ents:
            print(f"  WARN empty response for chunk at {i}", flush=True)
        for q in chunk:
            e = ents.get(q)
            out[q] = None if (e is None or e.get("missing") is not None) else e
        print(f"  {min(i+50, len(qids))}/{len(qids)}", flush=True)
        if i + 50 < len(qids):
            time.sleep(pause)
    return out


def label_of(ent):
    if not ent:
        return None
    return ((ent.get("labels") or {}).get("en") or {}).get("value")


def question(relation_label: str, subject_label: str) -> str:
    return f"What is the {relation_label} of {subject_label}?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--screen", default="evidence/screen_n3000.jsonl")
    ap.add_argument("--out", default="evidence/probe_sets.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- release index: relation -> candidate (subject, object) facts ----
    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            rows.append(r)
    by_rel = defaultdict(list)
    for r in rows:
        s, p, o = r.get("subject_id"), r.get("relation_id"), r.get("object_id")
        sl, rl, ol = r.get("subject"), r.get("relation"), r.get("object")
        if all(isinstance(x, str) for x in (s, p, o, sl, rl, ol)):
            by_rel[p].append((s, sl, o, ol, rl))
    print(f"{len(rows)} released rows; {len(by_rel)} relations indexed")

    # ---- edits: structurally valid rows only, so we test probes not edits ----
    screened = [json.loads(l) for l in open(args.screen, encoding="utf-8")
                if l.strip()]
    valid = [r for r in screened if r.get("operational_class") == "endpoint-valid"]
    print(f"{len(valid)} endpoint-valid candidate edits")
    rng.shuffle(valid)

    edits = []
    for r in valid:
        p = r["property_id"]
        if len(by_rel.get(p, [])) < 30:
            continue          # need a pool to draw fuzzy/random probes from
        edits.append(r)
        if len(edits) >= args.n:
            break
    print(f"selected {len(edits)} edits\n")

    # ---- choose probe subjects ----
    items = []
    for e in edits:
        s, p, o = e["subject_id"], e["property_id"], e["target_object_id"]
        sl = e.get("subject_label")
        pool = [c for c in by_rel[p] if c[0] != s and c[2] != o]
        if len(pool) < 10 or not isinstance(sl, str):
            continue
        rel_label = pool[0][4]

        # fuzzy: label-nearest subject, exactly the pipeline's argmax rule
        scored = sorted(pool,
                        key=lambda c: -SequenceMatcher(None, c[1].lower(),
                                                       sl.lower()).ratio())
        fz = scored[0]
        fz_sim = SequenceMatcher(None, fz[1].lower(), sl.lower()).ratio()

        # random: uniform draw from the same pool
        rd = pool[rng.randrange(len(pool))]
        rd_sim = SequenceMatcher(None, rd[1].lower(), sl.lower()).ratio()

        items.append({
            "edit_subject_id": s, "edit_subject_label": sl,
            "edit_property_id": p, "edit_relation_label": rel_label,
            "edit_object_id": o, "edit_object_label": e.get("object_label"),
            "fuzzy": {"subject_id": fz[0], "subject_label": fz[1],
                      "object_id": fz[2], "object_label": fz[3],
                      "similarity": round(fz_sim, 3)},
            "random": {"subject_id": rd[0], "subject_label": rd[1],
                       "object_id": rd[2], "object_label": rd[3],
                       "similarity": round(rd_sim, 3)},
        })
    print(f"{len(items)} items with usable fuzzy/random probes")

    # ---- verify probe facts + build the same-subject stratum ----
    need_claims = sorted({i["edit_subject_id"] for i in items}
                         | {i["fuzzy"]["subject_id"] for i in items}
                         | {i["random"]["subject_id"] for i in items})
    print(f"\nfetching claims for {len(need_claims)} entities "
          f"({(len(need_claims)+49)//50} requests)")
    claims = fetch(need_claims, "claims|labels")

    # same-subject probe: another property on the edited entity with exactly one
    # item-valued main snak, excluding the edited property
    extra_objs = set()
    for it in items:
        ent = claims.get(it["edit_subject_id"])
        it["same"] = None
        if not ent:
            continue
        cands = []
        for occ in occurrences(ent):
            if occ.role != "main" or occ.statement_property == it["edit_property_id"]:
                continue
            if occ.entity_type != "item" or occ.rank == "deprecated":
                continue
            a = answer_sets(ent, occ.statement_property)
            if len(a.primary) == 1:
                cands.append((occ.statement_property, occ.value_id))
        if cands:
            cands.sort()
            pick = cands[rng.randrange(len(cands))]
            it["same"] = {"subject_id": it["edit_subject_id"],
                          "subject_label": it["edit_subject_label"],
                          "property_id": pick[0], "object_id": pick[1],
                          "similarity": 1.0}
            extra_objs.add(pick[1])

    # labels for same-stratum objects and property labels
    need_labels = sorted(extra_objs)
    need_props = sorted({it["same"]["property_id"] for it in items if it["same"]})
    print(f"\nfetching {len(need_labels)} object labels")
    obj_labels = fetch(need_labels, "labels") if need_labels else {}
    print(f"fetching {len(need_props)} property labels")
    prop_labels = fetch(need_props, "labels") if need_props else {}

    # ---- verify each probe fact holds in current Wikidata ----
    def holds(subject_id, property_id, object_id):
        ent = claims.get(subject_id)
        if not ent:
            return False
        return object_id in answer_sets(ent, property_id).primary

    final = []
    stats = defaultdict(int)
    for it in items:
        rec = {k: v for k, v in it.items() if k not in ("fuzzy", "random", "same")}
        ok = {}
        for name in ("fuzzy", "random"):
            pr = it[name]
            if holds(pr["subject_id"], it["edit_property_id"], pr["object_id"]):
                ok[name] = {**pr, "property_id": it["edit_property_id"],
                            "relation_label": it["edit_relation_label"],
                            "question": question(it["edit_relation_label"],
                                                 pr["subject_label"]),
                            "answer": pr["object_label"]}
                stats[f"{name}-verified"] += 1
            else:
                stats[f"{name}-unverified"] += 1
        sm = it.get("same")
        if sm and holds(sm["subject_id"], sm["property_id"], sm["object_id"]):
            pl = label_of(prop_labels.get(sm["property_id"]))
            ol = label_of(obj_labels.get(sm["object_id"]))
            if pl and ol:
                ok["same"] = {**sm, "relation_label": pl, "object_label": ol,
                              "question": question(pl, sm["subject_label"]),
                              "answer": ol}
                stats["same-verified"] += 1
            else:
                stats["same-nolabel"] += 1
        else:
            stats["same-unverified"] += 1

        # keep only items where all three strata are available, so every
        # comparison is within-item and fully paired
        if len(ok) == 3:
            rec["probes"] = ok
            rec["edit_question"] = question(it["edit_relation_label"],
                                            it["edit_subject_label"])
            rec["edit_answer"] = it["edit_object_label"]

            # Counterfactual target. Editing a fact to the value it already has
            # cannot leak, so the edit must assert something the model does not
            # already believe. Drawn from the same relation's object pool and
            # required to differ from the true object and from every probe
            # answer, so that an appearance in a probe response is unambiguously
            # attributable to the edit.
            forbidden = {(rec["edit_object_label"] or "").lower()}
            forbidden |= {(ok[k]["answer"] or "").lower() for k in ok}
            pool_objs = [c[3] for c in by_rel[it["edit_property_id"]]
                         if isinstance(c[3], str)
                         and c[3].lower() not in forbidden]
            if not pool_objs:
                continue
            rec["counterfactual_answer"] = pool_objs[rng.randrange(len(pool_objs))]
            final.append(rec)

    print("\n=== probe verification ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:22s} {v}")
    print(f"\ncomplete paired items (all 3 strata): {len(final)}")

    if final:
        import statistics
        fz = [i["probes"]["fuzzy"]["similarity"] for i in final]
        rd = [i["probes"]["random"]["similarity"] for i in final]
        print(f"\nsubject-label similarity to the edit subject:")
        print(f"  fuzzy  mean {statistics.mean(fz):.3f}  median {statistics.median(fz):.3f}")
        print(f"  random mean {statistics.mean(rd):.3f}  median {statistics.median(rd):.3f}")
        print(f"  separation: {statistics.mean(fz)-statistics.mean(rd):+.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(final, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")

    if final:
        print("\n=== example item ===")
        e = final[0]
        print(f"  EDIT   : {e['edit_question']} -> {e['edit_answer']}")
        for k in ("fuzzy", "random", "same"):
            p = e["probes"][k]
            print(f"  {k:6s} : {p['question']} -> {p['answer']}   "
                  f"(sim {p['similarity']})")


if __name__ == "__main__":
    main()
