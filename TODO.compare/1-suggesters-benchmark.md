# Plan C1: benchmark the suggestion engine against the field

## Status

executed and re-executed under C6 (2026-09-21). Final frozen verdict:
**en — kotoshu #1 on all slices** (nonword 86.4/94.4/95.7 vs SymSpell
85.4 flat, Hunspell 78.5/93.6/95.1; realword 9.4% best-in-class).
**de — kotoshu #1 on top-3/top-5/realword** (91.1/92.4/11.1); top-1
trails SymSpell by 3pp on 79 pairs (C9: umlauts, vowel substitution,
frequency ties, transpositions). Reports: eval/reports/suggest-benchmark-{en,de}.json.

## Problem

The campaign has never measured the core product — the suggestion
engine — against the field. "SOTA" is currently a vibe. The real-word
arc only progressed because its gate was honest; the ranking engine
has no gate of its own.

## What

Benchmark harness `scripts/benchmark_suggesters.py` over the frozen
typo-corpus splits (eval/realword/{lang}.json pairs, plus the non-word
side where available): top-1/3/5 exact-match accuracy of the human
correction. Contenders:

- kotoshu (the gem's suggest path, en dictionary)
- Hunspell (ispell -a protocol, libreoffice/dictionaries en_US)
- SymSpell (symspellpy, our own frequency list as the dictionary)
- LanguageTool (public API on a bounded subsample, rate-limit labeled)
- the corpus's human correction as the oracle ceiling

Output: eval/reports/suggest-benchmark-{lang}.json (frozen evidence)
plus a markdown table in the plan. Honest verdict wherever it lands.

## Consumers

Plan C5 (per-language evals ride this harness); every future ranking
decision (edit-distance weights, rerank tiers) measures against this
baseline instead of opinion.
