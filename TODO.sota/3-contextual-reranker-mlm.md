# Plan S3: contextual MLM reranker — score, don't generate

## Status

researched and scoped (2026-09-22). The established pattern for real-word errors: plug each candidate into sentence context and score with a masked language model (Salazar et al., MLM scoring, pseudo-log-likelihood — foundational, cited 300+); SoftCorrect (arXiv Dec 2023) applies soft detection + MLM correction for ASR; Boosting Language Models for Real-Word Error Detection (Masanti et al., 2025) pushes detection quality with boosted LM ensembles. This is fundamentally different from the failed C2 cloze: rank EXISTING candidates by bidirectional context, never generate.

## Problem

Real-word errors ("there"→"their") are context-bound by definition. Our frequency+edit ranking is context-free; Hunspell's morphology beats us on es/fr realword. A context scorer over our own slate closes the class without FP risk (a reranker cannot invent candidates).

## What

1. Model: MiniLM-class multilingual encoder, int8 ONNX (plan S6), ~30-80M params, sliding window over sentence with the candidate slot masked.
2. Score every slate candidate via masked-slot pseudo-likelihood; blend with the existing ranking (SymSpell frequency order) — learnable 2-feature blend first, learned weights later.
3. Train/fine-tune on plan S1 synthetic context pairs (correct word in context → corrupted candidate) + clean-corpus contrastive pairs.
4. Gate: C1 harness realword slices per language; FP budget unchanged (rerank-only); latency budget ≤10ms/sentence on server tier (plan S6 lanes).
5. This replaces the C2 cloze artifact in the product surface (analyze/semantic path).

## Consumers

Real-word class everywhere; es/fr realword vs Hunspell; the browser demo's contextual scoring.
