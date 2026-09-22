# Plan S2: constrained seq2seq corrector — the cloze successor

## Status

researched and scoped (2026-09-22). The C2 ladder died because pseudo-likelihood on clean text never teaches correction. The field's answer: small encoder-decoders trained directly on (typo, correction) pairs — Low-Latency Spell Correction for Japanese Music Search (arXiv 2026, BART 3+3 layers, low latency); GEC SOTA shape (Qorib et al.: T5/GECToR-class minimal-edit systems; BEA-2019 combined F0.5 73.08); and the precision pattern N-best T5 + ASR-correction-with-LLMs (arXiv:2509.15095): CONSTRAIN the decoder's output space to the candidate slate. A constrained corrector cannot hallucinate — the over-correction failure that plagues LLM GEC (Fang et al. arXiv:2303.14342; Lin 2024; edit-level majority voting arXiv 2026) is structurally impossible.

## Problem

The real-word class (where we trail Hunspell on es/fr) needs a neural ranker that sees the whole word AND the correction relation, trained on actual error pairs, that never invents out-of-slate answers.

## What

1. Data: plan S1 synthetic pairs + frozen human splits (en 5000, es/fr/pt/ru, de) + LLM-distilled hard negatives (plan S5).
2. Model: compact encoder-decoder (BART-class, 3+3 layers, ~40-60M params), input = (typo, slate) serialized, output = the slate index / corrected word. Constrained decode = slate-classification head rather than free generation (stronger than beam-constrained, trivially ONNX).
3. Train on Modal A10G; export ONNX IR-10 int8 (plan S6); integrate as a SymSpellStrategy-slate reranker in the composite.
4. Gate: the frozen C1 harness, per language. FP budget: zero out-of-slate output by construction; measure slate-rerank top-1 lift.
5. Compare against S3 (they are complementary: S2 = word-relation modeling, S3 = context modeling).

## Consumers

The real-word class; es/fr top-1; the kotoshu-rs engine parity roadmap.
