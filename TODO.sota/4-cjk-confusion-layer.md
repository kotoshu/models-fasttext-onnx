# Plan S4: CJK confusion-set layer — pinyin + grapheme

## Status

researched and scoped (2026-09-22). CSC is a mature subfield with a fresh map: Chinese Spelling Correction survey (arXiv Feb 2025); C-LLM (ACL 2024, cited 27 — LLMs fail char-level CSC constraints; char-by-char checking fixes it); CSCD-NS (ACL 2024, cited 25 — 40k native-speaker samples, phonetic+graphemic error tags); PESC/RCSC (ACL 2024 — pinyin misspellings, SOTA on the realistic benchmark); DeCopy (2025 — pinyin-based detect-correct masking); PLOME (foundational pretraining with misspellings); WWR-CS (ACM 2025); QSpell 250K (ACL 2025 — large-scale Chinese query spell-correction data). Our substrate exists: IME layouts, variant-pure frequency lists, SymSpell-over-Han.

## Problem

Chinese errors are confusion-set phenomena (same-pinyin, near-pinyin, visual-lookalike) — edit distance over Han glyphs models NONE of these. ja kana-kanji conversion errors are the same class.

## What

1. Build per-variant confusion sets from open data: pinyin-syllable homophone tables + glyph-component similarity (the survey's taxonomy) for zh-Hans-CN/zh-Hant-TW/zh-Hant; romanization confusions for ja.
2. Wire confusion-set membership as a ranking feature in the composite (a confusion-set hit outranks an equal-distance non-hit) — no new model needed for v1.
3. Detect-correct masking (DeCopy pattern): pinyin-coverage detection flags candidate confusion regions before ranking.
4. Eval: CSCD-NS + RCSCB through the C1 harness (extend the harness for sentence-level CSC scoring: char-level precision/recall); train S2/S3 heads on QSpell-250K-class data.
5. C-LLM's constraint insight folds into S2: character-by-character (word-by-word) slate classification, never free generation.

## Consumers

zh-Hans-CN/zh-Hant-TW/zh-Hant-HK suggestion quality; ja suggestions; the CJK dictionary arc's quality half.
