# Plan 15: real-word detection — eval pairs with context, confusion tables, calibration

## Status: Phase 0 executed — v0 scorer FAILED the gate; Phase 1 (trained ctx-LM) is the open work

Phase 0 findings (docs/realword-detection-design.md carries the
numbers): the DL≤1 table covers only 15.6% of real-word instances
(the dominant classes are distance-2); the cosine-margin scorer is
WORSE than a raw frequency prior at every FP anchor; and the
skipgram rung is dead upstream — the shipped cc.*.300 output
matrices are all zeros (verified at true offsets), so the
conditional scorer must be TRAINED, not extracted. Phase 1: n-gram
ctx-LM (en first) + confusion table v2 (bounded d=2 + phonetic),
gated per the design doc before the engine plans unblock.

## Problem

Detection is vocabulary membership everywhere in the ecosystem
(gem `semantic_analyzer.rb` `next if valid_word?`; rs check path the
same). Real-word errors — an in-vocab word that is wrong in context
(en "I want to each rice"; zh 再/再-class homophone selections) — are
invisible to the product, and every model we ship (tiers, buckets,
KTM1, fluency) only ranks corrections for words ALREADY flagged. The
missing artifact layer: real-word eval pairs WITH sentence context, a
per-language confusion-pair table (which in-vocab words are plausible
mis-selections for which), and a calibration harness that measures
recall AND clean-text false-positive rate for a margin-threshold
detector before any engine work is attempted.

## What

- `scripts/extract_realword_pairs.py`: re-extracts from the raw
  GitHub Typo Corpus jsonl (reusing `fetch_corpus._word_pair`), keeps
  edits whose TYPO SIDE is also in the language's full-tier vocab
  (the real-word class), and emits `eval/realword/{lang}.json` with
  {typo, correction, context sentence, count}. The corrected
  sentences become the clean-text FP probe set.
- `scripts/build_confusion_tables.py`: enumerates in-vocab confusion
  pairs per language — edit-distance ≤ 1 plus keyboard-adjacent
  substitutions plus shared-phonetic-class — via deletion-neighborhood
  hashing (no O(n²)); emits `eval/confusion/{lang}.json`. CJK
  homophone-sourced tables (pinyin/kana readings over the tier vocab)
  are a follow-on phase, not this plan's scope.
- `scripts/eval_realword_detection.py`: the calibration probe — scores
  the in-context word vs each confusion candidate (v0 scorer: cosine
  of the word's tier vector against its context neighbors, the same
  math as rs `CosineReranker`), sweeps margin thresholds, reports the
  recall/FP curve. Gates the engine plans: if no threshold clears the
  agreed operating point (design doc records the curve), the v0
  scorer is insufficient and the doc's scorer ladder (skipgram
  output-matrix margin, then n-gram LM) decides the escalation.
- `docs/realword-detection-design.md`: the design record — mechanism,
  scorer ladder, calibration numbers, per-language FP budget,
  rollout phases, artifact formats.

## Consumers

Gem plan 146 (analyzer detection branch, opt-in API) and rs plan 07
(native scoring) consume the confusion-table artifact format and the
frozen thresholds this plan produces. Registry shipping of confusion
tables as a resource type is deferred until the engine plans land —
tables ship as eval artifacts first.
