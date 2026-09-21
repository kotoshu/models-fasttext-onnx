# Plan C8: cross-language #1 loop

## Status

pending on C6 + C7. The C1 harness already takes --lang. This plan is the fleet loop: for every language with a dictionary + Phase-0 pairs, extract suggest splits, run the harness, freeze the report, and only call the language "quality-gated" when kotoshu nonword top-1 ≥ max(Hunspell, SymSpell) and real-word top-1 ≥ max(field). First wave after C6 lands: en, de, es, fr, pt, ru. CJK (ja/zh-*) blocked on dictionary acquisition (C3).

## Problem

One-off en/de benches do not prove #1 across the board. Each language has its own keyboard, frequency list, and error distribution.

## What

1. Script `scripts/run_suggest_fleet.py` — for each lang in the wave: extract pairs if missing, ensure frequency list installed, run benchmark_suggesters, write eval/reports/suggest-benchmark-{lang}.json.
2. Gate table in this plan updated per run.
3. Failures file a per-language follow-up (dictionary gap, frequency gap, layout gap) rather than a silent miss.

## Consumers

Release quality bar; C5; marketing claims that must stay honest.
