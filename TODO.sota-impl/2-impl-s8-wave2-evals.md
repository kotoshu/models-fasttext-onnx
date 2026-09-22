# Impl S8: wave-2 realistic evals (graph node S8)

## Status

RUNNING (2026-09-22). Splits frozen (12 languages × suggest2-*).
kelly#6 merged (nl/pl); gem PR #229 (KELLY 18 codes). Field dicts:
it/nl/pl added (wooorm). Harness: --split2 lane + wave2 report names.
Benchmarks executing in background (kotoshu lane is slow on big
dictionaries — hours; reports land in eval/reports/*-wave2.json). Deliverables: ≥2,000 nonword + ≥200 realword pairs per language for en de es fr pt ru it nl pl + zh-Hans-CN zh-Hant-TW ja (class-tagged); kelly lists for nl/pl (missing); hunspell field dicts for new languages; full C1 harness runs per language (field lanes + kotoshu); frozen reports; public repo republish; verdict table with class columns.

## Gate

G-S8: all languages frozen ≥2,000 nonword; wave-1 verdicts re-earned or revised HONESTLY.

## Consumers

The re-earned "#1 across the board" claim; S2/S3/S9 gates.
