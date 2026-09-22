# Impl S4: CJK confusion layer v1 (graph node S4)

## Status

pending. Deliverables: pinyin homophone tables built from CC-CEDICT (group by syllable); per-variant confusion JSON published to the dictionaries repo; Ruby ranking feature (confusion-set hit outranks equal-distance non-hit); zh IME-class eval via S1 generator; CSCD-NS integration if license-clean.

## Gate

G-S4: zh IME-class top-1 lifts vs C8 baseline; char-level P/R harness extension for CSCD-NS (if dataset lands).

## Consumers

zh-Hans-CN/zh-Hant-TW/zh-Hant-HK/ja suggestion quality.
