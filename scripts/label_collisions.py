"""Label ambiguity inside the release itself -- no API required.

The pipeline drops QIDs after extraction and keys everything downstream on labels
(`filter_ambiguous` groups by (subject_label, relation_label); locality and
multi-hop match on label strings). The release still carries QIDs, so the
ambiguity the pipeline was exposed to can be measured directly:

  * how many distinct QIDs share one subject label;
  * how often one label maps to several entities *within the same interval*,
    which is where the label-keyed grouping actually operated;
  * whether the "unique answer" guarantee survives at QID level, given that it
    was enforced at label level.

Any leakage here is evidence that label-keyed joins could not have preserved
identity, independent of the Wikidata-side audit.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wbe_audit.stats import wilson  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/label_collisions.json")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("data/raw/wiki_big_edit_*.json")):
        base = os.path.basename(f)
        for i, r in enumerate(json.load(open(f, encoding="utf-8"))):
            rows.append((base, i, r))
    print(f"{len(rows)} released rows\n")

    subj_label_to_ids = defaultdict(set)
    subj_id_to_labels = defaultdict(set)
    obj_label_to_ids = defaultdict(set)
    for _, _, r in rows:
        s, sl = r.get("subject_id"), r.get("subject")
        o, ol = r.get("object_id"), r.get("object")
        if isinstance(sl, str) and s:
            subj_label_to_ids[sl].add(s)
            subj_id_to_labels[s].add(sl)
        if isinstance(ol, str) and o:
            obj_label_to_ids[ol].add(o)

    def spread(d, name):
        sizes = Counter(len(v) for v in d.values())
        tot = len(d)
        multi = sum(v for k, v in sizes.items() if k > 1)
        print(f"=== {name} ===")
        print(f"  distinct keys                 : {tot}")
        print(f"  keys mapping to >1 entity     : {multi}  ({100*multi/tot:.2f}%)")
        worst = sorted(d.items(), key=lambda kv: -len(kv[1]))[:8]
        for k, v in worst:
            print(f"    {len(v):4d} QIDs share the label {k!r}")
        print()
        return {"total": tot, "multi": multi}

    st1 = spread(subj_label_to_ids, "subject label -> distinct QIDs (whole release)")
    st2 = spread(obj_label_to_ids, "object label -> distinct QIDs (whole release)")

    # rows affected, not just keys affected
    amb_rows = sum(1 for _, _, r in rows
                   if isinstance(r.get("subject"), str)
                   and len(subj_label_to_ids[r["subject"]]) > 1)
    n = len(rows)
    print("=== rows whose subject label is shared by >1 QID in the release ===")
    print(f"  {wilson(amb_rows, n).pct()}\n")

    amb_obj_rows = sum(1 for _, _, r in rows
                       if isinstance(r.get("object"), str)
                       and len(obj_label_to_ids[r["object"]]) > 1)
    print("=== rows whose object label is shared by >1 QID in the release ===")
    print(f"  {wilson(amb_obj_rows, n).pct()}\n")

    # the label-keyed uniqueness guarantee, re-checked at QID level
    per_interval = defaultdict(lambda: defaultdict(set))
    per_interval_lab = defaultdict(lambda: defaultdict(set))
    for base, _, r in rows:
        s, p, o = r.get("subject_id"), r.get("relation_id"), r.get("object_id")
        sl, rl, ol = r.get("subject"), r.get("relation"), r.get("object")
        if s and p and o:
            per_interval[base][(s, p)].add(o)
        if isinstance(sl, str) and isinstance(rl, str) and isinstance(ol, str):
            per_interval_lab[base][(sl, rl)].add(ol)

    tot_g = multi_g = 0
    for base, groups in per_interval.items():
        tot_g += len(groups)
        multi_g += sum(1 for v in groups.values() if len(v) > 1)
    tot_l = multi_l = 0
    for base, groups in per_interval_lab.items():
        tot_l += len(groups)
        multi_l += sum(1 for v in groups.values() if len(v) > 1)

    print("=== 'unique answer' guarantee, per interval ===")
    print(f"  by (subject_label, relation_label): "
          f"{multi_l}/{tot_l} groups have >1 object  ({100*multi_l/tot_l:.3f}%)")
    print(f"  by (subject_id, relation_id)      : "
          f"{multi_g}/{tot_g} groups have >1 object  ({100*multi_g/tot_g:.3f}%)")
    print("  the filter was applied on labels, so leakage at QID level is expected")
    print("  only if labels and QIDs disagree.\n")

    # one QID carrying several different labels across the release
    multi_lab = {k: v for k, v in subj_id_to_labels.items() if len(v) > 1}
    print("=== one subject QID carrying >1 distinct label ===")
    print(f"  {len(multi_lab)} of {len(subj_id_to_labels)} QIDs "
          f"({100*len(multi_lab)/len(subj_id_to_labels):.2f}%)")
    for k, v in list(sorted(multi_lab.items(), key=lambda kv: -len(kv[1])))[:8]:
        print(f"    {k}: {sorted(v)[:5]}")

    json.dump({"subject_label_keys": st1, "object_label_keys": st2,
               "rows_ambiguous_subject": amb_rows,
               "rows_ambiguous_object": amb_obj_rows, "n_rows": n,
               "groups_multi_by_id": multi_g, "groups_total_by_id": tot_g,
               "groups_multi_by_label": multi_l, "groups_total_by_label": tot_l},
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
