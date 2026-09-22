# Plan S8: wave-2 realistic eval — MulTypo taxonomy, per-class reporting

## Status

researched and scoped (2026-09-22). Two anchors: MulTypo (arXiv:2510.09536) supplies a language-specific, keyboard-grounded error taxonomy for 12+ languages with code; Evaluating Robustness of LLMs Against Typos (arXiv Oct 2025, cited 9) supplies the evaluation methodology (multilingual, realistic-error robustness). The Multilingual GEC Shared Task (ACL 2024/2025) is the community venue to stay aligned with.

## Problem

First-wave splits for es/fr/pt/ru are 71-304 pairs (within-noise gaps at top-1), and human-corpus slices underrepresent diacritic-omission and IME error classes that our C9 fold and IME layouts target.

## What

1. Generate wave-2 splits with plan S1's engine at MulTypo-equivalent class rates: 2,000+ pairs per language across en/es/fr/pt/ru/de/it/nl/pl + zh variants (IME class) + ja.
2. Freeze with per-class metadata (substitution-adjacent, transposition, double-letter, diacritic, IME, realword) so reports break accuracy down BY CLASS — the fleet verdict table gains class columns.
3. Add field LLM lanes as ROBUSTNESS baselines (small bounded samples, clearly labelled offline batch, not product lanes) so our claims sit next to the field's strongest systems.
4. Publish the regenerated splits + reports through the public suggesters-benchmark repo (corpus-free, generator-reproducible).
5. Re-freeze C8's table; the "#1 across the board" claim gets re-earned on wave-2 data.

## Consumers

The honest-claims law; C6/C8/C9 verdicts; the public benchmark repo's credibility.
