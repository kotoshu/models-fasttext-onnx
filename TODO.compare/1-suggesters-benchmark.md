# Plan C1: benchmark the suggestion engine against the field

## Status

in-progress (2026-09-21, owner: "do all of these properly")

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
