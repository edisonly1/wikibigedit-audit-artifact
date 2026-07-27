"""Score the adjudication against the frozen classifier.

Scope: this is a consistency / face-validity check, not blinded double-coding. A
single adjudicator applied the class definitions to each packet's structured
evidence; where the verdict and classifier agree, the mechanical rule implements the
written definition on that case. It does not establish inter-annotator reliability;
external blinded adjudication is left as a limitation.

Adjudications below are keyed by audit_id, each a (fine_class, note). Coarse families
collapse the fine classes into the five that matter for the headline, so a fine-label
quibble doesn't count as a headline disagreement.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

PACKETS = sys.argv[1] if len(sys.argv) > 1 else "evidence/phaseA_packets.json"

COARSE = {
    "endpoint-valid": "VALID",
    "main-supported-nonunique": "VALID_NONUNIQUE",
    "role-erased-qualifier": "ROLE_ERASED",
    "role-erased-reference": "ROLE_ERASED",
    "property-absent-on-subject": "IDENTITY_ABSENT",
    "object-qid-misattached": "OBJECT_META",
    "object-genuinely-wrong": "OBJECT_META",
    "object-absent-unresolved": "OBJECT_META",
}

# audit_id -> {"fine": human fine class, "note": reasoning}
ADJ = {
 "wiki_big_edit_20240501_20240601.json:7058:Q706679:P407:Q37283709": {
   "fine": "object-absent-unresolved",
   "note": "No main P407; released Catalan-QID absent entirely; P407 present only "
           "in a non-main role for other objects. Invalid, object-metadata family."},
 "wiki_big_edit_20240220_20240301.json:1159:Q184774:P427:Q55936638": {
   "fine": "object-qid-misattached",
   "note": "Actual taxonomic type Q222951 is ALSO labelled 'Gallus' (sim 1.0); "
           "released Q55936638 is a homonym QID. Textbook misattachment."},
 "wiki_big_edit_20240401_20240501.json:94493:Q12641094:P407:Q6654": {
   "fine": "endpoint-valid",
   "note": "rho_WBE main set is exactly {Q6654} = released object. Valid."},
 "wiki_big_edit_20240320_20240401.json:26399:Q7980710:P407:Q5378163": {
   "fine": "object-qid-misattached",
   "note": "Actual value Q1860 (canonical English) shares the label 'English' with "
           "released Q5378163. Homonym QID."},
 "wiki_big_edit_20240501_20240601.json:47570:Q5402737:P131:Q761927": {
   "fine": "object-qid-misattached",
   "note": "Actual P131 Q1486 (Buenos Aires city) shares label with released "
           "Q761927. Homonym QID."},
 "wiki_big_edit_20240501_20240601.json:82099:Q14875133:P407:Q5378163": {
   "fine": "object-absent-unresolved",
   "note": "No main P407; object absent; no resolvable main value. Invalid, "
           "object-metadata family."},
 "wiki_big_edit_20240401_20240501.json:114157:Q17597771:P407:Q784704": {
   "fine": "object-absent-unresolved",
   "note": "No main P407; object absent. Same shape as the two above."},
 "wiki_big_edit_20240501_20240601.json:66234:Q7995447:P5817:Q11639308": {
   "fine": "property-absent-on-subject",
   "note": "P5817 absent in EVERY role; station has no state-of-use statement. "
           "Identity/absence failure."},
 "wiki_big_edit_20240301_20240320.json:16635:Q3400799:P17:Q16275867": {
   "fine": "property-absent-on-subject",
   "note": "P17 country absent in every role on the journal Pouvoirs. Identity "
           "failure. (Released France-QID is itself an odd Q16275867 homonym.)"},
 "wiki_big_edit_20240601_20240620.json:15478:Q2848902:P972:Q110271769": {
   "fine": "role-erased-reference",
   "note": "'Adlib Museum' appears in a REFERENCE snak on a P6379 statement, not "
           "as a main fact. Role erasure."},
 "wiki_big_edit_20240320_20240401.json:40329:Q17342319:P31:Q16127605": {
   "fine": "main-supported-nonunique",
   "note": "Object Q16127605 is one of two P31 main values ({Q16127605, Q532}). "
           "Valid under rho_all, non-unique under rho_WBE."},
 "wiki_big_edit_20240401_20240501.json:50166:Q56266913:P123:Q28863790": {
   "fine": "role-erased-reference",
   "note": "CYSTAT appears only in reference snaks (many host props). Role erasure. "
           "Snapshot resolved to a 2021 revision (entity unedited since)."},
 "wiki_big_edit_20240401_20240501.json:99665:Q15080048:P123:Q10543423": {
   "fine": "role-erased-reference",
   "note": "'Kaparna' appears in reference snaks only. Role erasure."},
 "wiki_big_edit_20240201_20240220.json:24826:Q17151937:P1437:Q15916502": {
   "fine": "role-erased-qualifier",
   "note": "'guilty plea' is a qualifier on a P1595 (charge) statement, flattened "
           "into a main 'plea' fact. Role erasure."},
 "wiki_big_edit_20240620_20240701.json:12447:Q4802028:P768:Q988730": {
   "fine": "role-erased-qualifier",
   "note": "Electoral district is a qualifier on a P3602 (candidacy) statement. "
           "Role erasure."},
 "wiki_big_edit_20240601_20240620.json:42969:Q6465114:P421:Q6760": {
   "fine": "endpoint-valid",
   "note": "rho_WBE main set exactly {Q6760}. Valid."},
 "wiki_big_edit_20240401_20240501.json:45920:Q4771058:P197:Q6960081": {
   "fine": "object-genuinely-wrong",
   "note": "Actual adjacent station is Toun (Q8189950), a different station "
           "(sim 0.625 < 0.85). Genuinely wrong. (rho_WBE also lists the subject "
           "itself, a self-loop oddity.)"},
 "wiki_big_edit_20240601_20240620.json:28806:Q5689642:P2378:Q23548": {
   "fine": "role-erased-qualifier",
   "note": "NASA is an 'issued by' qualifier on a P528 (catalog code) statement. "
           "Clean role-erasure example."},
 "wiki_big_edit_20240320_20240401.json:29603:Q10916553:P31:Q811979": {
   "fine": "main-supported-nonunique",
   "note": "Object is one of three P31 main values. Valid under rho_all, "
           "non-unique under rho_WBE."},
 "wiki_big_edit_20240601_20240620.json:59637:Q15962907:P31:Q7725634": {
   "fine": "object-genuinely-wrong",
   "note": "QID differs (actual Q3331189 translation / Q49084 short story, sim "
           "0.333). Rule => genuinely-wrong. SEMANTIC CAVEAT: 'literary work' is "
           "arguably a defensible superclass answer, so the text supervision is "
           "not obviously wrong even though the QID is. Flagged."},
 "wiki_big_edit_20240501_20240601.json:84842:Q16008601:P137:Q124156216": {
   "fine": "endpoint-valid",
   "note": "rho_WBE main set exactly {Q124156216}. Valid."},
 "wiki_big_edit_20240401_20240501.json:64649:Q10324471:P735:Q2855657": {
   "fine": "object-genuinely-wrong",
   "note": "Given name is 'Manuel' (Q11113719), released says 'Victor' (sim 0.16). "
           "Genuinely wrong / different value."},
 "wiki_big_edit_20240501_20240601.json:66102:Q7984863:P5817:Q55654238": {
   "fine": "property-absent-on-subject",
   "note": "P5817 absent in every role on West Concord station. Identity failure."},
 "wiki_big_edit_20240320_20240401.json:8238:Q1353694:P106:Q2519376": {
   "fine": "main-supported-nonunique",
   "note": "Object is one of two P106 main values. Non-unique under rho_WBE."},
}


def main() -> None:
    packets = json.load(open(PACKETS, encoding="utf-8"))
    fine_agree = coarse_agree = scored = 0
    disagreements = []
    for p in packets:
        aid = p["audit_id"]
        adj = ADJ.get(aid)
        if not adj or not adj["fine"]:
            print(f"  MISSING adjudication for {aid}")
            continue
        scored += 1
        model_fine = p["model_class"]
        human_fine = adj["fine"]
        if model_fine == human_fine:
            fine_agree += 1
        if COARSE.get(model_fine) == COARSE.get(human_fine):
            coarse_agree += 1
        else:
            disagreements.append((aid, model_fine, human_fine, adj["note"]))

    print(f"scored packets            : {scored}")
    print(f"fine-class agreement      : {fine_agree}/{scored} "
          f"({100*fine_agree/scored:.1f}%)")
    print(f"coarse-family agreement   : {coarse_agree}/{scored} "
          f"({100*coarse_agree/scored:.1f}%)")

    print("\ncoarse family distribution (model):")
    for k, v in Counter(COARSE.get(p["model_class"]) for p in packets
                        if ADJ.get(p["audit_id"], {}).get("fine")).most_common():
        print(f"  {k:18s} {v}")

    if disagreements:
        print("\ncoarse-family disagreements:")
        for aid, mf, hf, note in disagreements:
            print(f"  {aid}\n    model={mf} human={hf}\n    {note}")
    else:
        print("\nno coarse-family disagreements.")

    print("\nfine-label notes worth carrying into the paper:")
    for aid, adj in ADJ.items():
        if "CAVEAT" in adj.get("note", "") or "SEMANTIC" in adj.get("note", ""):
            print(f"  {aid}: {adj['note']}")


if __name__ == "__main__":
    main()
