# Plan 16: Phase 1 — the trained ctx-LM, confusion table v2, and the gate

## Status: executed — gate FAILED for the bigram rung; evidence frozen, ladder escalated

Table v2 passed its half (coverage 15.6% -> 62.2%, after fixing the
v1 identity-variant index bug that had silently missed every
insert/delete pair). The bigram ctx-LM failed its half decisively:
argmax capability exceeds the bar (65.5% true-top at tau=0) but clean
and error margin distributions overlap (70.5% FP at tau=0; 5.1%
recall at the 1% FP operating point); frequency-proximity filters and
min-support do not separate them. Numbers + the three scorer bugs
(conditioning, polarity, vacuous-zero candidates) recorded in
docs/realword-detection-design.md Phase 1 section; frozen evidence in
eval/realword/en.probe.ctxlm.json. The n-gram rung is dead — next
rung is a neural context scorer, an owner decision. Plans 146/07 stay
blocked.

## Problem

Phase 0 (plan 15) proved the shipped machinery cannot DETECT real-word
errors: DL≤1 confusion coverage is 15.6% (distance-2 dominates), the
cosine margin scores 8.1% true-top at 1% FP (below the 16.1%
context-free frequency floor), and the cc.*.300 output matrices are
zero — the conditional scorer must be trained. Nothing above changes
until two artifacts exist and pass one gate.

## What

- **Table v2** (`scripts/build_confusion_tables.py` grows `--band`):
  bounded DL≤2 pairs — at least one side in the top-30k frequency
  band — via the 2-deletion (band side) × 1-deletion (vocab side)
  index meet with DL≤2 verification. Sources recorded per pair
  (`dl1`/`dl2`); corpus pairs stay excluded (they are the eval set).
- **Corpus**: ~1M en sentences from a license-clean source (Leipzig
  Corpora Collection Wikipedia 1M preferred; provenance + license
  recorded in the artifact manifest). Choice of corpus is the design
  doc's; scaling beyond en is owner-gated.
- **ctx-LM trainer** (`scripts/train_ctx_lm.py`): unigram + bigram
  counts over the corpus, lowercased fetch_corpus._clean_token
  tokens, backoff α=0.4 (stupid backoff), int8-per-row quantized
  bigram table keyed by hashed bigrams + the unigram table, emitted
  as `models/en/fasttext.en.ctx.npz` (tables) with a JSON manifest
  (counts, source, license, backoff, quantization). ONNX packaging
  happens in the unblock PRs — the gate decides whether the format
  matters, not vice versa.
- **The gate** (`scripts/eval_realword_detection.py --scorer ctxlm`,
  table v2, SAME frozen Phase-0 split): eligible-token FP ≤ 1% AND
  covered-instance true-top ≥ 60%. PASS unblocks gem plan 146 + rs
  plan 07 (status flips); FAIL records the curve and escalates the
  ladder (trigram context, then the Phase-5 transformer rung) —
  gates are never weakened.
- **Tests** (repo convention, unittest): table-v2 bounded-distance
  correctness + band rule on synthetic vocab; trainer determinism +
  quantization round-trip + backoff math on a tiny synthetic corpus.

## Consumers

The design doc gains the Phase-1 numbers. On PASS: gem 146 / rs 07
unblock with the artifact format frozen; registry resource type is
their follow-up. On FAIL: the escalation note replaces the unblock.
