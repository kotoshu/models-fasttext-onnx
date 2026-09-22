# Plan C11: research sweep — 2023-2026 techniques worth adopting

## Status

researched (2026-09-22, owner: "Research more techniques! arxiv 2023-2026"). The sweep's headline: **LLMs alone are NOT the answer for our class of problem** — ACL 2024 realistic CSC benchmarks show LLMs losing on precision/F0.5 to tuned traditional pipelines, and the industry direction (Grammarly) is compact 1B on-device models, not API LLMs. The techniques worth adopting, ranked by leverage:

1. **Supervised seq2seq corrector** — Low-Latency Spell Correction for Japanese Music Search (arXiv 2026, Garg et al.): compact BART 3-encoder/3-decoder, LOW latency, direct (typo, correction) supervision. This is the successor to the FAILED cloze transformer (C2): the cloze failed because pseudo-likelihood on clean text doesn't teach correction; a seq2seq trained ON error pairs does. ONNX-exportable. Direct consumer: the real-word class where we trail Hunspell on es/fr.
2. **Phonetic word embeddings** — Zouhar et al., ACL 2024 ("Phonetic Word Embeddings and Tasks They Facilitate", cited 14): pronunciation-similar embeddings upgrade our soundex-class PhoneticStrategy into a multilingual phonetic channel — the es/fr realword gap is accent/phonetic class, exactly this.
3. **CJK confusion sets + pinyin masking** — CSC survey (arXiv Feb 2025); Decopy (pinyin-based error masking, 2025); PMDRSpell/WWR-CS (confusion-set fusion, 2024-2025); CSCD-NS (40k native-speaker eval, ACL 2024). Our IME layouts + frequency channel are the substrate; confusion-set scoring and pinyin masking are the next zh quality layer, with CSCD-NS as the honest eval.
4. **Realistic multilingual typo benchmarks** — "Evaluating Robustness of LLMs Against Typos" (arXiv Oct 2025) — methodology for wave-2 splits (realistic error distributions per language, not uniform edits).
5. **Validated: hybrid beats LLM-only** — Turhan et al. 2025 (contextual models beat context-free tools) + ACL 2024 (LLMs lose on precision) → our architecture (candidate generation + language-aware ranking + contextual layer) is the published consensus shape.

## Problem

The engine must adopt 2023-2026 techniques, not 2016-era ones, to stay #1.

## What

- C11a: train the compact seq2seq corrector on our frozen suggest splits + corpus-mined pairs (Modal A10G), ONNX IR-10, gate on the same C1 harness. THE priority — replaces the failed cloze line with a design that directly optimizes the metric.
- C11b: phonetic embedding table for the top-20 fleet languages (train or adopt released vectors), wired as an additional ranking feature.
- C11c: zh confusion-set layer (pinyin + grapheme), evaluated on CSCD-NS.
- C11d: wave-2 split generation using the robustness paper's error taxonomy.

## Consumers

The real-word frontier; zh/ja quality; wave-2 evals; the C1 harness.
