# Artifact and reproducibility guide

This repository is the anonymous artifact for *A Role-Aware Audit of WikiBigEdit*.
It contains the code, the audited data, and the scripts that generate every table
and figure. 

The list below maps the reviewer-relevant artifacts to their locations.

## 1. Sampled rows, inclusion status, labels, revisions, corrections

`release/sampled_rows_manifest.jsonl` (one record per sampled row, 1,500 rows)
carries, for each row:

- `audit_id`, `subject_id`, `property_id`, `object_id`, `source_interval`,
  `source_row_index` (sampled row identity);
- `resolvable`, `unresolvable_reason` (inclusion status; 1,458 resolvable, 42 not);
- `operational_class`, `endpoint_verdict_rho_WBE`, `valid_rho_all`,
  `valid_rho_truthy` (audit labels);
- `pinned_revision_id`, `revision_timestamp` (exact revision the verdict is
  computed against);
- `corrected_object_id` (the 337 misattached-identifier corrections) and
  `resolved_actual_object_id` (the entity's actual main value where resolved).

Regenerate: `python scripts/make_artifact_manifest.py`.

## 2. Repaired resource and evidence

- `release/wikibigedit_repaired.jsonl` — three-track repair set with per-row
  provenance and corrected identifiers; `release/repaired_manifest.json` — coverage.
- `evidence/hist_n1500.jsonl` / `hist_n1500.reclassified.jsonl` — the audit before
  and after object reclassification.
- `evidence/extraction_verification_merged.jsonl` — independent re-parse used for
  validation (§Validation).
- `evidence/locality_*.jsonl`, `released_exp_*.jsonl`, `param_exp_ftl.jsonl`,
  `rome_exp.jsonl` — downstream-experiment records (per-item probes, answers,
  leakage/locality flags).

## 3. Extraction trace replay

Traces are not stored as static files because they are reproducible: given a pinned
`revision_id`, `src/wbe_audit/replay.py` re-fetches the exact stored page text and
replays the released 200-character scanner deterministically, aligning serialized
marker occurrences one-to-one with structural snaks. See
`scripts/test_replay.py` and `scripts/test_segment_wiki.py` for worked traces.

## 4. Random seeds

| use | seed |
|---|---|
| audit sampling (uniform SRS over rows) | 20260722 |
| relation-clustered bootstrap | 20260722 |
| probe-set construction, downstream experiments | 20260723 |
| sign-flip permutation test | 20260723 |
| extraction-verification subsample | 20260724, 20260725 |

All seeds are fixed in the scripts; no result depends on an unseeded draw. Bootstrap
uses 4,000 replicates; the permutation test uses 20,000 Rademacher sign vectors
(Appendix A of the paper).

## 5. Model prompts and decoding parameters

Defined in `src/wbe_audit/llm.py` (Ollama) and `src/wbe_audit/param_edit.py` (HF).

- **System prompt** (factual QA): "You answer factual questions with a short noun
  phrase and nothing else. Output only the answer, at most a few words." (full text
  in `llm.py:SYSTEM`).
- **Edit-context template** and **scope-gate prompt**: `scripts/run_locality_experiment.py`
  (`with_fact`, `SCOPE_SYSTEM`, `SCOPE_TEMPLATE`, selected by
  `scripts/tune_scope_gate.py`).
- **Decoding (Ollama)**: temperature 0, top_k 1, top_p 1, seed 0, num_predict 24,
  repeat_penalty 1.0 (greedy, deterministic).
- **Decoding (HF)**: `do_sample=False`, `num_beams=1`, `max_new_tokens=24`.
- **Determinism protocol**: the first evaluation of any prompt is discarded and the
  second is used; justified in `scripts/repeat_stability.py` (instability is
  confined to the first call, 0/120 later-call mismatches per model).
- Backbones: `qwen2.5:7b`, `llama3.1:8b` (Q4_K_M, Ollama); `Qwen2.5-1.5B-Instruct`
  (fp32, HF) for parameter editing.

## 6. Editor hyperparameters

- **FT-L** (`param_edit.py:EditConfig`): edit layer 12, Adam lr $10^{-3}$, 30 steps,
  $L_\infty$ norm constraint $5\times10^{-3}$; selected by
  `scripts/check_edit_locality.py`. Edit success 52.7%.
- **ROME** (`rome.py:RomeConfig`): edit layer 12, value optimisation lr 0.5, 40
  steps, weight decay $10^{-3}$, clamp $8\times$ baseline norm; key covariance $C$
  ridge $\lambda = 0.05\cdot\mathrm{mean(diag)}$, estimated from 40,005 key
  positions over released questions. Validated so the edited layer maps
  $k^\*\to v^\*$ to relative error $10^{-7}$ with exact restore
  (`scripts/validate_rome.py`). Edit success 83.0%.

## 7. Scripts for every table and figure

| paper element | script(s) |
|---|---|
| Table 1 (prevalence, Wilson + rel-clu) | `cluster_bootstrap.py`, `perclass_relclu.py` |
| Table 2 (endpoint sensitivity, 4-way split) | `sensitivity_table.py` |
| Table 3 (as-deployed DiD: leak/loc/WBE) | `analyse_released_exp.py`, `param_conditional.py` |
| Table 4 (matched reconstruction) | `reconstruction_analysis.py`, `reconstruction_v2.py`; ROME row: `run_rome_matched.py`, `analyse_rome_matched.py` |
| Table 5 (corruption of known facts) | `corruption_table.py` (subgroup counts: `baseline_correct_subgroup.py`) |
| Table 6 (repaired-record examples) | `repaired_examples.py` |
| Figure 1 (pipeline) | drawn in TikZ (see the paper PDF) |
| Figure 2 (role-erasure trace) | drawn in TikZ (see the paper PDF) |
| Figure 3 (prevalence bars) | `make_figures.py` |
| Figure 4 (timestamp sweep) | `make_figures.py` (data: `timestamp_calibration.py`) |
| Figure 5 (locality provenance) | `make_figures.py` (data: `locality_provenance.py`) |
| Figure 6 (leakage bars) | `make_figures2.py` |
| Figure 7 (dose-response) | `make_figures2.py`; trend test: `dose_trend_test.py` |
| ITT vs conditional-on-success DiD | `itt_conditional.py` |
| Validation PPV/FOR (confusion matrix) | `verify_extraction.py`, `score_validation.py` |
| Interval robustness, unconditional split | `interval_and_unconditional.py` |
| Locality-probe provenance (98.9%) | `locality_provenance.py`, `locality_similarity_null.py` |
| Classifier unit tests | `tests/test_semantics.py` |

## Environment

Python 3.12; `requests`, `numpy`, `pandas`, `scipy`, `scikit-learn`,
`statsmodels`, `matplotlib`, `torch`, `transformers`, `gensim`. Ollama for the 7B/8B
backbones; a 12 GB GPU suffices for the 1.5B parameter-editing arm. Wikidata
access is via the public MediaWiki API (historical revision content cannot be
batched; a serial crawl at roughly 10 requests/minute is assumed). Snapshots are
cached under `data/cache/` (gitignored) so re-runs are free.

## Licensing

Code in this repository is released under the MIT License (`LICENSE`). The audited
and repaired data derive from WikiBigEdit (`lukasthede/WikiBigEdit`, released under
its own terms) and from Wikidata (CC0); downstream users must honour those upstream
licenses. No new human-subjects data is collected.
