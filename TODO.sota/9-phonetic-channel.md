# Plan S9: phonetic embedding channel — pronunciation-aware ranking

## Status

researched and scoped (2026-09-22). Phonetic Word Embeddings and Tasks They Facilitate (Zouhar et al., ACL 2024, cited 14) provides embeddings that encode pronunciation similarity and demonstrates misspelling correction as a direct task; our PhoneticStrategy is still soundex-class (English-centric, coarse).

## Problem

The es/fr real-word misses are accent/phonetic class (é/è swaps, silent letters, diacritic morphology) — exactly where Hunspell's phonet tables win. Grapheme-to-phoneme confusion is also the ja romaji and de umlaut-adjacent error surface.

## What

1. Adopt/train multilingual phonetic embeddings (Zouhar's method or released vectors) for the top-20 fleet languages; export as a compact ONNX lookup (the fleet law: no runtime API deps).
2. Wire as a ranking feature in the composite: phonetic-similarity(candidate, typo) joins distance + frequency + keyboard + (S2/S3 when armed).
3. Eval: per-class C1 harness runs (wave-2 splits carry the class tags) — the accent class is the target metric; es/fr realword vs Hunspell is the headline.
4. Order: after S1 (data) and alongside S2 (the corrector consumes the same feature).

## Consumers

es/fr realword; de diacritic residuals (C9); ja romaji class; S2's slate features.
