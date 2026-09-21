# Plan C3: dictionary depth audit — the measured gap map

## Status

acquisition executed (2026-09-21) — the CJK gap is CLOSED for
suggestions. frequency-list-kelly#3 (merged) ships variant-pure CJK
frequency lists: zh-Hans-CN 90,389 (CC-CEDICT-validated), zh-Hant
55,144 (script-generic), zh-Hant-TW 61,239 (moedict/Taiwan-MOE
validated), ja 84,770 (wordfreq). dictionaries PR kotoshu/dictionaries#4
ships the matching spelling wordlists. The gem (PR #226) indexes them —
SymSpell suggests for CJK with zero new code ("我扪"→"我们", "日今語"→
"日本語" verified live). zh-Hant-HK remains an explicit gap (no HK
frequency source; documented, not faked). Remaining follow-up: plain-
text spelling resolution in the gem (CJK wordlists are words.txt, not
Hunspell aff+dic — ResourceManager resolves Hunspell only today), and
the C8 fleet evals for CJK once detection is wired.

## Problem

57 languages have models; non-word detection is dictionary-bound, and
nobody has measured which languages lack Hunspell aff+dic coverage.
The zh-Hant dictionary gap is already on the books but unquantified.

## What

Cross the 57 registry languages against the kotoshu/dictionaries repo
manifest (which carries per-language aff+dic pins): three sets —

1. full coverage (dictionary + model)
2. model-only languages (detection limited to dictionary-less paths)
3. dictionary-only languages (models absent — the fleet holdbacks)

Output: eval/reports/dictionary-coverage.json + the gap table in this
plan, with the zh-Hant/TW/HK dictionary acquisition path named (edu.tw
and MOE-derived open wordlists, converted to Hunspell or unified-dict
form). Acquisition itself is follow-up work — this plan measures.

## Consumers

The fleet retrain queue; the zh variants' spell-check completeness;
the C5 per-language evals (a language can only be benchmarked where a
dictionary exists).
