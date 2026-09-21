# Plan C3: dictionary depth audit — the measured gap map

## Status

in-progress (2026-09-21)

## Problem

57 languages have models; non-word detection is dictionary-bound, and
nobody has measured which languages lack Hunspell aff+dic coverage.
The zh-Hant dictionary gap is already on the books but unquantified.

## What

Cross the 57 registry languages against the kotoshu/dictionaries repo
manifest (which carries per-language aff+dic pins): three sets —

1. full coverage (dictionary + model)
2. model-only languages (detection limited to dictionary-less paths)
3. dictionary-only languages (models absent — the fleet holdbacks)

Output: eval/reports/dictionary-coverage.json + the gap table in this
plan, with the zh-Hant/TW/HK dictionary acquisition path named (edu.tw
and MOE-derived open wordlists, converted to Hunspell or unified-dict
form). Acquisition itself is follow-up work — this plan measures.

## Consumers

The fleet retrain queue; the zh variants' spell-check completeness;
the C5 per-language evals (a language can only be benchmarked where a
dictionary exists).
