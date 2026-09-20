# Plan D13: real-word detection beyond English — the demo's headline feature, fleet-wide pattern

## Status

executed with an honest failed-gate (2026-09-20) — SHIPPED: the German bigram context tables (102.3M wiki tokens -> models/de/fasttext.de.ctx.npz, 53 MB, under the git limit), per-language confusion tables for de/es/fr/pt/ru frozen as committed evidence (previously eval/confusion/ was ignored entirely - even en.json was local-only), fetch_wiki_corpus.py's two real path bugs fixed, and the demo's German confusion staging. BLOCKED BY GATE, RECORDED: German DETECTION in the demo does not ship - the de real-word evidence is 97 edits (en: 54,073), a ±1 bigram window cannot carry German verb-final syntax (diagnosis: the decisive dass-clause collocation sits 2+ tokens out; two of three draft fixtures were bad German), and fitting thresholds until fixtures pass is exactly the gate-weakening docs/realword-detection-design.md forbids. Detection lands per language when plan 15/16's calibration harness passes it. The ja/ko confusion tables were built and DISCARDED (mean degree 229/57 - the subword-starvation artifact; noise, not evidence) pending the segmentation-aware derivation.

## Problem

The directive quotes the demo sentence, and the demo's headline feature
— real-word (confusion-set) detection — is English-only: the explorer
loads all 57 languages, but detection has en's confusion table and en's
bigram context tables alone. Full 57-language detection is plan 15/16's
calibration arc (per-language frozen thresholds demand per-language
real-word eval sets); the honest increment now is the languages whose
evidence pipeline already exists, executed end to end and gated.

## What

1. Inventory the per-language evidence on disk (confusion tables,
   real-word eval pairs, corpus availability).
2. For German — the fleet-proven corpus path — train the bigram
   context tables (scripts/train_ctx_lm.py --lang de), derive the
   confusion table from the same evidence pipeline that produced
   eval/confusion/en.json, and freeze a fixture validation set of
   German real-word errors (seit/seid, das/dass, wider/wieder class).
3. Extend the demo: detection becomes language-aware (en + de), with
   the conservative dual gate per language and honest labeling of the
   calibration state.
4. Ship the de artifacts through the standard surfaces (plain git,
   raw-host mirrors, registry-consistent), keep file sizes under the
   100 MiB git limit.
5. Validate: Python-side fixture gates per language BEFORE the browser
   run, then the headless browser E2E for de detection.
6. The per-language recipe is then documented for the remaining
   languages — each lands when its evidence and calibration pass, per
   the design doc's gate, not before.

## Consumers

The demo's users; plan 15/16 inherits the per-language pipeline.
