# A Role-Aware Audit of WikiBigEdit — code and data

Anonymous artifact for the AKBC 2026 (EMNLP workshop) submission *When String
Matching Stands In for a Knowledge Base: A Role-Aware Audit of WikiBigEdit*.

This repository contains the audit code, the sampled-row manifest and labels, the
three-track repair set, the downstream-experiment records, and the scripts that
generate every table and figure in the paper. See **`ARTIFACT.md`** for the full
reproducibility guide (sampled rows, seeds, prompts, decoding parameters, editor
hyperparameters, and the table/figure-to-script map).

## Layout

```
src/wbe_audit/     audit engine: revision access, role/rank semantics, scanner
                   replay, classification, statistics, LLM + parameter editors
scripts/           data fetch, audits, screens, calibration, experiments, analysis
tests/             classifier unit tests
evidence/          audit and experiment outputs (JSONL)
release/           repaired resource + per-row sampled-row manifest
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_release.py          # download the benchmark into data/raw/
python tests/test_semantics.py           # classifier unit tests (offline)
python scripts/release_stats.py          # descriptive stats over the release
```

Wikidata access uses the public MediaWiki API. Historical revision content cannot
be batched, so the audit runs single-threaded at roughly ten requests per minute;
snapshots are cached under `data/cache/` (gitignored). The
parameter-editing arm needs a single 12 GB GPU and the Ollama backbones
(`qwen2.5:7b`, `llama3.1:8b`).

## Reproducing specific results

`ARTIFACT.md` lists, for each table and figure, the exact script and input file.
For example:

```bash
python scripts/cluster_bootstrap.py          # Table 1 intervals + inclusion flow
python scripts/sensitivity_table.py          # Table 2
python scripts/reconstruction_v2.py          # Table 4 + permutation tests
python scripts/make_figures.py               # Figures 3-5
python scripts/make_figures2.py              # Figures 6-7
```

## License

Code under the MIT License (`LICENSE`). Audited and repaired data derive from
WikiBigEdit and Wikidata (CC0) and remain subject to their upstream licenses.
