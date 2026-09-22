# Plan C8: cross-language #1 loop

## Status

first wave executed (2026-09-22) — en/es/fr/pt/ru/de frozen in
eval/reports/suggest-benchmark-{lang}.json (field lanes: LibreOffice/
wooorm Hunspell dictionaries + the SAME published kelly lists the gem
indexes; kotoshu lane: the Ruby engine, KOTOSHU_BACKEND=ruby):

| lang | n(nonword) | kotoshu top1/top3/top5 | field SymSpell top1 | field Hunspell top1 | kotoshu nonword #1 |
|------|-----------|------------------------|---------------------|---------------------|--------------------|
| en   | 2000 | 86.4 / 94.4 / 95.7 | 85.4 | 78.5 | YES all slices |
| es   | 71   | 76.1 / 87.3 / 87.3 | 74.7 | 59.2 | YES (realword: Hunspell 25.6 leads) |
| fr   | 119  | 68.1 / 84.0 / 84.9 | 65.6 | 61.3 | YES (top3 ties Hunspell; realword: Hunspell 12.1 leads) |
| ru   | 304  | 74.0 / 86.8 / 88.2 | 75.0 | 70.7 | top3/top5 YES; top1 −1.0pp (within n=304 noise) |
| pt   | 124  | 66.1 / 79.0 / 82.3 | 67.7 | 62.9 | top5 YES; top1 −1.6pp (n=124 noise) |
| de   | 79   | 70.9 / 91.1 / 92.4 | 73.4 | 54.4 | top3/top5 YES; top1 −2.5pp (C9) |

Real-word class: kotoshu leads en/de/pt/ru, Hunspell leads es/fr —
the known frontier (realword is context-bound; Hunspell's morphology
helps on accent swaps). Fleet wave 2: bigger pair splits (the corpus
is thin for es/fr/pt/ru), C9 residual patterns, ja/zh once CJK
detection wiring lands.

Frequency sources shipped for the wave: frequency-list-kelly#4
(es/fr/pt + ru blend); gem PR c8-fleet-languages (KELLY_LANGUAGES).

## Problem

One-off en/de benches do not prove #1 across the board. Each language has its own keyboard, frequency list, and error distribution.

## What

1. Script `scripts/run_suggest_fleet.py` — for each lang in the wave: extract pairs if missing, ensure frequency list installed, run benchmark_suggesters, write eval/reports/suggest-benchmark-{lang}.json.
2. Gate table in this plan updated per run.
3. Failures file a per-language follow-up (dictionary gap, frequency gap, layout gap) rather than a silent miss.

## Consumers

Release quality bar; C5; marketing claims that must stay honest.
