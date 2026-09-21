# Plan C5: per-language suggestion quality evals

## Status

executed for en + de (2026-09-21) — de: 79 nonword + 72 realword pairs (the corpus is thin for de, labeled). de nonword top-1: SymSpell 73.4% (wiki unigram dictionary) > Hunspell 54.4% > kotoshu 50.6%; real-word: kotoshu 8.3% best-in-class (Hunspell 5.6%). The pattern mirrors en: context-free engines are strong on nonwords, the real-word class is the open frontier. Future languages: one command each (extract pairs -> run harness) wherever a dictionary + Phase-0 pairs exist.

## Problem

The fleet has coverage gates (wordfreq ≥99%) but not quality gates:
nothing measures whether the corrections a language's model+dictionary
produce are RIGHT. The real-word arc's honesty came from frozen eval
splits; suggestion quality needs the same.

## What

The C1 harness takes --lang: given eval/realword/{lang}.json pairs
and that language's dictionary, it measures top-1/3/5 exact-match for
every contender. First run: en (12,648 pairs) and de (270 pairs —
thin, labeled as such). The harness, not a bespoke script, is the
product: every future language audit is one command.

Frozen per-language reports join eval/reports/. A language enters the
"quality-gated" set only when its eval runs.

## Consumers

The fleet retrain queue (quality gates join coverage gates); C3's
dictionary audit (a model-only language with no dictionary cannot be
quality-gated — the dependency is explicit).
