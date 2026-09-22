# Plan S1: synthetic typo engine — MulTypo-class, layout-grounded

## Status

researched and scoped (2026-09-22). The field's consensus (Misspellings in NLP: A Survey, arXiv Oct 2025; Karpukhin et al., cited 134): training on synthetic noise is THE robustness lever, and the current best generator is MulTypo (arXiv:2510.09536, cited 9, code github.com/cisnlp/multypo) — multilingual typo generation grounded in language-specific keyboard layouts and typing behavior. We already own the hard part MulTypo builds on: a 21-layout registry (QWERTZ/AZERTY/dubeolsik/Arabic/Chinese-IME...) with dual-layout scoring. What we lack is the generator that turns layouts into realistic error distributions.

## Problem

Our eval splits for es/fr/pt/ru are tiny (71-304 pairs) and our training data for any neural corrector is limited to the thin GitHub Typo Corpus slices. Uniform random edits do not match human error distributions; models trained on them underperform.

## What

1. `scripts/generate_typos.py`: per-language typo synthesizer over our keyboard-layout registry — substitution weights = key adjacency (layout-aware, dual-layout aware per plan C7), plus transposition/double-letter/diacritic-omission classes with corpus-fitted rates, plus IME-class errors for zh/ja (pinyin-syllable confusions, kana conversion slips).
2. Emit two artifacts per language: (a) realistic eval splits (wave-2, plan S8), (b) unlimited (typo, correction) training pairs from the frequency full_lists and clean corpora.
3. Validate realism: distribution of generated error classes vs the human GitHub Typo Corpus splits (KL divergence report); a generator that fails this check does not ship.
4. Consider contributing our layout registry upstream to MulTypo (they cover 12+ languages; we have IME layouts they lack).

## Consumers

S2 (constrained corrector) and S3 (contextual reranker) training data; S8 wave-2 splits; C1/C8 harness realism upgrade.
