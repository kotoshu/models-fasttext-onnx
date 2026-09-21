# Plan C1: benchmark the suggestion engine against the field

## Status

executed (2026-09-21); re-executed under C6 (frequency SymSpell + language_code + ranked:true composite, gem PR kotoshu/kotoshu#226). de nonword: top-1 70.9%, top-3 86.1%, top-5 91.1% — top-3 and top-5 #1 across the field (Hunspell 54.4%/74.7%/77.2%, field SymSpell 73.4% on all slices — TOP mode returns one candidate). en bench in flight. C9 (TODO.compare/9) tracks the residual top-1 gap to take sole #1. real-word top-1 best-in-class at 9.1% but the class is context-bound and hard for everyone (Hunspell 7.4%, SymSpell 4.4%) - quantifying the context-scorer's product value. LanguageTool's 0 is a shape artifact (sentence-checker contract vs isolated words). Evidence: eval/reports/suggest-benchmark-en.json + the frozen splits (en.suggest-{nonword,realword}.json, 2000 pairs/class). En-route fixes: symspellpy's load_dictionary defaults to a SPACE separator; canonical pair extraction must reuse fetch_corpus.extract_pairs.

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
