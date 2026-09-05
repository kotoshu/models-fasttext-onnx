#!/usr/bin/env python3
"""Deterministic keyboard-aware multilingual typo generator.

Replaces run_eval.py's original ASCII-lowercase-only typo model. Two script
families, one API: ``make_typo(rng, word, lang) -> str | None``.

Layout model (MULTYPO-style, arxiv 2005.01158) for de/en/es/fr/pt/ru:
- Physical key maps are mirrored from the kotoshu gem's own layout files
  (lib/kotoshu/keyboard/layouts/{qwerty,qwertz,azerty,jcuken}.rb), which are
  the source of truth: QWERTY (en/es/pt), QWERTZ incl. the umlaut/eszett
  keys (de), AZERTY incl. the ù key (fr), JCUKEN (ru). Key coordinates are
  the gem's [row, col] grids; "neighbor" = Manhattan distance 1 or 2, with
  distance-1 keys weighted 3x distance-2 keys (near keys slip more often).
- The gem layouts only carry unshifted base keys, so dead-key/diacritic
  slips are added as a documented per-language supplement: an accented
  letter (á, é, ß, ñ, ç, ё, ...) is a neighbor of its base letter and of
  its diacritic siblings (spelling-level slips, weight 1, or 3 toward the
  base letter from the accented side). German umlaut keys ä/ö/ü/ß are real
  QWERTZ keys and get physical neighbors in addition.

Plan 77 coverage expansion (it nl pl uk tr cs sv el hu ro da vi ca):
- Latin-script languages reuse the gem's QWERTY/QWERTZ physical grids with
  a per-language diacritic supplement, exactly the es/pt/fr pattern above.
- Three languages cannot reuse a gem grid and get curated national grids
  here (the gem has no tr/uk/el layout files yet — when it grows them,
  mirror the coordinates from there and delete the local copies):
  Turkish-Q (tr: ı ğ ü ş i ö ç are real keys), Ukrainian ЙЦУКЕН (uk:
  і ї є ґ replace the Russian-only keys), Greek phonetic (el: the standard
  EL layout is phonetically equivalent; positions simplified, accents via
  the diacritic supplement). Curated, not gem-mirrored — same honesty
  rule as the CJK confusion sets.

Character-confusion model for ja/ko/zh (no key adjacency applies):
- Small curated visual/phonetic confusion pairs per language (see
  _JA_PAIRS / _KO_PAIRS / _ZH_PAIRS below): kanji/kana lookalikes for ja,
  jamo pairs (vowel mergers ㅐ/ㅔ, y-vowel slips, shape flips, fortis /
  aspirated consonant confusion) applied *inside* Hangul syllables for ko,
  shape-similar hanzi for zh. These sets are CURATED from well-known
  confusion lists, not learned from data — they are a stand-in until a
  confusion matrix can be measured from a real CJK typo corpus.
- Adjacent-character transposition is the script-neutral fallback (and the
  only op when no character of the word is in a confusion pair).

Operations for layout languages: substitute (neighbor of the substituted
char) / insert (neighbor of the adjacent char) / delete / transpose, with
weights 0.40/0.20/0.20/0.20. For CJK: confusion-substitute 0.65 /
transpose 0.35.

Determinism: every draw comes from the caller-supplied seeded generator
(np.random.default_rng([42, crc32(language)]) in the eval harness); this
module never touches global RNG state, so make_typo is a pure function of
(rng state, word, lang). Vocabulary membership is NOT checked here — the
caller (run_eval.top1_agreement_metric) rejects probes whose typo is out
of vocabulary, exactly as before.
"""

from __future__ import annotations

import sys

import numpy as np

TYPO_ATTEMPTS = 50

OP_WEIGHTS_LAYOUT = (("substitute", 0.40), ("insert", 0.20), ("delete", 0.20), ("transpose", 0.20))
OP_WEIGHTS_CJK = (("substitute", 0.65), ("transpose", 0.35))

WEIGHT_D1 = 3.0  # physical neighbor at Manhattan distance 1
WEIGHT_D2 = 1.0  # physical neighbor at Manhattan distance 2
WEIGHT_ALTERNATE = 1.0  # diacritic sibling (dead-key slip)
WEIGHT_BASE = 3.0  # accented char -> base letter (dropped diacritic)

# ---------------------------------------------------------------------------
# Keyboard layouts — mirrored verbatim from the kotoshu gem
# (lib/kotoshu/keyboard/layouts/*.rb). char -> [row, col].
# ---------------------------------------------------------------------------

_QWERTY = {
    "`": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "=": (0, 12),
    "q": (1, 0), "w": (1, 1), "e": (1, 2), "r": (1, 3), "t": (1, 4), "y": (1, 5),
    "u": (1, 6), "i": (1, 7), "o": (1, 8), "p": (1, 9), "[": (1, 10), "]": (1, 11), "\\": (1, 12),
    "a": (2, 0), "s": (2, 1), "d": (2, 2), "f": (2, 3), "g": (2, 4), "h": (2, 5),
    "j": (2, 6), "k": (2, 7), "l": (2, 8), ";": (2, 9), "'": (2, 10),
    "z": (3, 0), "x": (3, 1), "c": (3, 2), "v": (3, 3), "b": (3, 4), "n": (3, 5),
    "m": (3, 6), ",": (3, 7), ".": (3, 8), "/": (3, 9),
}

_QWERTZ = {
    "^": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "ß": (0, 11), "´": (0, 12),
    "q": (1, 0), "w": (1, 1), "e": (1, 2), "r": (1, 3), "t": (1, 4), "z": (1, 5),
    "u": (1, 6), "i": (1, 7), "o": (1, 8), "p": (1, 9), "ü": (1, 10), "+": (1, 11),
    "a": (2, 0), "s": (2, 1), "d": (2, 2), "f": (2, 3), "g": (2, 4), "h": (2, 5),
    "j": (2, 6), "k": (2, 7), "l": (2, 8), "ö": (2, 9), "ä": (2, 10),
    "y": (3, 0), "x": (3, 1), "c": (3, 2), "v": (3, 3), "b": (3, 4), "n": (3, 5),
    "m": (3, 6), ",": (3, 7), ".": (3, 8), "-": (3, 9),
}

_AZERTY = {
    "`": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), ")": (0, 11), "=": (0, 12),
    "a": (1, 0), "z": (1, 1), "e": (1, 2), "r": (1, 3), "t": (1, 4), "y": (1, 5),
    "u": (1, 6), "i": (1, 7), "o": (1, 8), "p": (1, 9), "^": (1, 10), "$": (1, 11),
    "q": (2, 0), "s": (2, 1), "d": (2, 2), "f": (2, 3), "g": (2, 4), "h": (2, 5),
    "j": (2, 6), "k": (2, 7), "l": (2, 8), "m": (2, 9), "ù": (2, 10),
    "w": (3, 0), "x": (3, 1), "c": (3, 2), "v": (3, 3), "b": (3, 4), "n": (3, 5),
    ",": (3, 6), ";": (3, 7), ":": (3, 8), "!": (3, 9),
}

_JCUKEN = {
    "ё": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "=": (0, 12),
    "й": (1, 0), "ц": (1, 1), "у": (1, 2), "к": (1, 3), "е": (1, 4), "н": (1, 5),
    "г": (1, 6), "ш": (1, 7), "щ": (1, 8), "з": (1, 9), "х": (1, 10), "ъ": (1, 11),
    "ф": (2, 0), "ы": (2, 1), "в": (2, 2), "а": (2, 3), "п": (2, 4), "р": (2, 5),
    "о": (2, 6), "л": (2, 7), "д": (2, 8), "ж": (2, 9), "э": (2, 10),
    "я": (3, 0), "ч": (3, 1), "с": (3, 2), "м": (3, 3), "и": (3, 4), "т": (3, 5),
    "ь": (3, 6), "б": (3, 7), "ю": (3, 8), ".": (3, 9),
}

# Turkish-Q (standard Turkish layout; ı ğ ü ş i ö ç are real keys).
_TR_Q = {
    "\"": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "*": (0, 11), "-": (0, 12),
    "q": (1, 0), "w": (1, 1), "e": (1, 2), "r": (1, 3), "t": (1, 4), "y": (1, 5),
    "u": (1, 6), "ı": (1, 7), "o": (1, 8), "p": (1, 9), "ğ": (1, 10), "ü": (1, 11),
    "a": (2, 0), "s": (2, 1), "d": (2, 2), "f": (2, 3), "g": (2, 4), "h": (2, 5),
    "j": (2, 6), "k": (2, 7), "l": (2, 8), "ş": (2, 9), "i": (2, 10),
    "z": (3, 0), "x": (3, 1), "c": (3, 2), "v": (3, 3), "b": (3, 4), "n": (3, 5),
    "m": (3, 6), "ö": (3, 7), "ç": (3, 8), ".": (3, 9),
}

# Ukrainian ЙЦУКЕН (standard Ukrainian layout; і ї є ґ replace the
# Russian-only ы ъ э ё keys at the same positions).
_UK_JCUKEN = {
    "'": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "=": (0, 12),
    "й": (1, 0), "ц": (1, 1), "у": (1, 2), "к": (1, 3), "е": (1, 4), "н": (1, 5),
    "г": (1, 6), "ш": (1, 7), "щ": (1, 8), "з": (1, 9), "х": (1, 10), "ї": (1, 11), "ґ": (1, 12),
    "ф": (2, 0), "і": (2, 1), "в": (2, 2), "а": (2, 3), "п": (2, 4), "р": (2, 5),
    "о": (2, 6), "л": (2, 7), "д": (2, 8), "ж": (2, 9), "є": (2, 10),
    "я": (3, 0), "ч": (3, 1), "с": (3, 2), "м": (3, 3), "и": (3, 4), "т": (3, 5),
    "ь": (3, 6), "б": (3, 7), "ю": (3, 8), ".": (3, 9),
}

# Greek phonetic mnemonic grid (standard EL layout is phonetically
# equivalent; accents are dead-key typed and handled by the supplement).
_EL_PHONETIC = {
    "`": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "=": (0, 12),
    ";": (1, 0), "σ": (1, 1), "ε": (1, 2), "ρ": (1, 3), "τ": (1, 4), "υ": (1, 5),
    "θ": (1, 6), "ι": (1, 7), "ο": (1, 8), "π": (1, 9),
    "α": (2, 0), "δ": (2, 1), "φ": (2, 2), "γ": (2, 3), "η": (2, 4), "ξ": (2, 5),
    "κ": (2, 6), "λ": (2, 7),
    "ζ": (3, 0), "χ": (3, 1), "ψ": (3, 2), "ω": (3, 3), "β": (3, 4), "ν": (3, 5), "μ": (3, 6),
}

# ---------------------------------------------------------------------------
# Plan 83 batch 2 grids. Same honesty rule as wave 1: curated national grids
# for scripts the gem has no layout file for, verified against the standard
# layout documentation (kbdlayout.info driver tables for the Windows
# national layouts, Culmus SI-1452 for Hebrew, ISIRI 9147 renderings for
# Persian, Arabic 101). When the gem grows native layout files, mirror the
# coordinates from there and delete the local copies.
# ---------------------------------------------------------------------------

# Arabic 101 (standard IBM PC Arabic layout).
_AR_101 = {
    "ذ": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "=": (0, 12),
    "ض": (1, 0), "ص": (1, 1), "ث": (1, 2), "ق": (1, 3), "ف": (1, 4), "غ": (1, 5),
    "ع": (1, 6), "ه": (1, 7), "خ": (1, 8), "ح": (1, 9), "ج": (1, 10), "د": (1, 11),
    "ش": (2, 0), "س": (2, 1), "ي": (2, 2), "ب": (2, 3), "ل": (2, 4), "ا": (2, 5),
    "ت": (2, 6), "ن": (2, 7), "م": (2, 8), "ك": (2, 9), "ط": (2, 10),
    "ئ": (3, 0), "ء": (3, 1), "ؤ": (3, 2), "ر": (3, 3), "لا": (3, 4), "ى": (3, 5),
    "ة": (3, 6), "و": (3, 7), "ز": (3, 8), "ظ": (3, 9),
}

# Persian standard (ISIRI 9147).
_FA_STD = {
    "`": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "=": (0, 12),
    "ض": (1, 0), "ص": (1, 1), "ث": (1, 2), "ق": (1, 3), "ف": (1, 4), "غ": (1, 5),
    "ع": (1, 6), "ه": (1, 7), "خ": (1, 8), "ح": (1, 9), "ج": (1, 10), "چ": (1, 11),
    "ش": (2, 0), "س": (2, 1), "ی": (2, 2), "ب": (2, 3), "ل": (2, 4), "ا": (2, 5),
    "ت": (2, 6), "ن": (2, 7), "م": (2, 8), "ک": (2, 9), "گ": (2, 10),
    "ظ": (3, 0), "ط": (3, 1), "ز": (3, 2), "ر": (3, 3), "ذ": (3, 4), "د": (3, 5),
    "پ": (3, 6), "و": (3, 7), "ۀ": (3, 8), ".": (3, 9),
}

# Hebrew standard SI-1452 (verified against the Culmus layout table; the
# five final-letter forms are real keys on this layout).
_HE_SI1452 = {
    "`": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "=": (0, 12),
    "/": (1, 0), "'": (1, 1), "ק": (1, 2), "ר": (1, 3), "א": (1, 4), "ט": (1, 5),
    "ו": (1, 6), "ן": (1, 7), "ם": (1, 8), "פ": (1, 9), "[": (1, 10), "]": (1, 11),
    "ש": (2, 0), "ד": (2, 1), "ג": (2, 2), "כ": (2, 3), "ע": (2, 4), "י": (2, 5),
    "ח": (2, 6), "ל": (2, 7), "ך": (2, 8), "ף": (2, 9), ",": (2, 10),
    "ז": (3, 0), "ס": (3, 1), "ב": (3, 2), "ה": (3, 3), "נ": (3, 4), "מ": (3, 5),
    "צ": (3, 6), "ת": (3, 7), "ץ": (3, 8), ".": (3, 9),
}

# Bulgarian BDS 5237:1978 (KBDBUL.DLL).
_BG_BDS = {
    ")": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "$": (0, 11), "€": (0, 12),
    "ы": (1, 0), "у": (1, 1), "е": (1, 2), "и": (1, 3), "ш": (1, 4), "щ": (1, 5),
    "к": (1, 6), "с": (1, 7), "д": (1, 8), "з": (1, 9), "ц": (1, 10),
    "ь": (2, 0), "я": (2, 1), "а": (2, 2), "о": (2, 3), "ж": (2, 4), "г": (2, 5),
    "т": (2, 6), "н": (2, 7), "в": (2, 8), "м": (2, 9), "ч": (2, 10),
    "ю": (3, 0), "й": (3, 1), "ъ": (3, 2), "э": (3, 3), "ф": (3, 4), "х": (3, 5),
    "п": (3, 6), "р": (3, 7), "л": (3, 8), "б": (3, 9),
}

# Serbian Cyrillic (ЉЊЕРТЗУИОПШ / АСДФГХЈКЛЧЋ / ЏЦВБНМ standard).
_SR_CYR = {
    "`": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "'": (0, 11), "+": (0, 12),
    "љ": (1, 0), "њ": (1, 1), "е": (1, 2), "р": (1, 3), "т": (1, 4), "з": (1, 5),
    "у": (1, 6), "и": (1, 7), "о": (1, 8), "п": (1, 9), "ш": (1, 10), "ђ": (1, 11),
    "а": (2, 0), "с": (2, 1), "д": (2, 2), "ф": (2, 3), "г": (2, 4), "х": (2, 5),
    "ј": (2, 6), "к": (2, 7), "л": (2, 8), "ч": (2, 9), "ћ": (2, 10), "ж": (2, 11),
    "џ": (3, 0), "ц": (3, 1), "в": (3, 2), "б": (3, 3), "н": (3, 4), "м": (3, 5),
    ",": (3, 6), ".": (3, 7), "-": (3, 8),
}

# Macedonian (KBDMAC.DLL; like Serbian with ѕ at т-position, ѓ/ќ letters,
# з moved to the bottom row).
_MK_CYR = {
    "~": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "=": (0, 12),
    "љ": (1, 0), "њ": (1, 1), "е": (1, 2), "р": (1, 3), "т": (1, 4), "ѕ": (1, 5),
    "у": (1, 6), "и": (1, 7), "о": (1, 8), "п": (1, 9), "ш": (1, 10), "ђ": (1, 11),
    "а": (2, 0), "с": (2, 1), "д": (2, 2), "ф": (2, 3), "г": (2, 4), "х": (2, 5),
    "ј": (2, 6), "к": (2, 7), "л": (2, 8), "ч": (2, 9), "ћ": (2, 10), "ж": (2, 11),
    "з": (3, 0), "џ": (3, 1), "ц": (3, 2), "в": (3, 3), "б": (3, 4), "н": (3, 5),
    "м": (3, 6), ",": (3, 7), ".": (3, 8), "/": (3, 9),
}

# Mongolian Cyrillic (KBDMON.DLL; distinct from Russian JCUKEN).
_MN_CYR = {
    "+": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "е": (0, 11), "щ": (0, 12),
    "ф": (1, 0), "ц": (1, 1), "у": (1, 2), "ж": (1, 3), "э": (1, 4), "н": (1, 5),
    "г": (1, 6), "ш": (1, 7), "ү": (1, 8), "з": (1, 9), "к": (1, 10), "ъ": (1, 11),
    "й": (2, 0), "ы": (2, 1), "б": (2, 2), "ө": (2, 3), "а": (2, 4), "х": (2, 5),
    "р": (2, 6), "о": (2, 7), "л": (2, 8), "д": (2, 9), "п": (2, 10),
    "я": (3, 0), "ч": (3, 1), "ё": (3, 2), "с": (3, 3), "м": (3, 4), "и": (3, 5),
    "т": (3, 6), "ь": (3, 7), "в": (3, 8), "ю": (3, 9),
}

# Armenian phonetic (KBDRME.DLL eastern Armenian; the western variant
# differs only in two keys and shares the same physical grid).
_HY_PHONETIC = {
    "`": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "=": (0, 12),
    "խ": (1, 0), "ւ": (1, 1), "է": (1, 2), "ր": (1, 3), "տ": (1, 4), "ե": (1, 5),
    "ը": (1, 6), "ի": (1, 7), "ո": (1, 8), "պ": (1, 9), "չ": (1, 10), "ջ": (1, 11),
    "ա": (2, 0), "ս": (2, 1), "դ": (2, 2), "ֆ": (2, 3), "ք": (2, 4), "հ": (2, 5),
    "ճ": (2, 6), "կ": (2, 7), "լ": (2, 8), "թ": (2, 9), "փ": (2, 10),
    "զ": (3, 0), "ց": (3, 1), "գ": (3, 2), "վ": (3, 3), "բ": (3, 4), "ն": (3, 5),
    "մ": (3, 6), "շ": (3, 7), "ղ": (3, 8), "ծ": (3, 9),
}

# Georgian national layout (KBDGEO.DLL).
_KA_NATIONAL = {
    "„": (0, 0), "!": (0, 1), "?": (0, 2), "№": (0, 3), "§": (0, 4), "%": (0, 5),
    ":": (0, 6), ".": (0, 7), ";": (0, 8), ",": (0, 9), "/": (0, 10), "–": (0, 11), "=": (0, 12),
    "ღ": (1, 0), "ჯ": (1, 1), "უ": (1, 2), "კ": (1, 3), "ე": (1, 4), "ნ": (1, 5),
    "გ": (1, 6), "შ": (1, 7), "წ": (1, 8), "ზ": (1, 9), "ხ": (1, 10), "ც": (1, 11),
    "ფ": (2, 0), "ძ": (2, 1), "ვ": (2, 2), "თ": (2, 3), "ა": (2, 4), "პ": (2, 5),
    "რ": (2, 6), "ო": (2, 7), "ლ": (2, 8), "დ": (2, 9), "ჟ": (2, 10),
    "ჭ": (3, 0), "ჩ": (3, 1), "ყ": (3, 2), "ს": (3, 3), "მ": (3, 4), "ი": (3, 5),
    "ტ": (3, 6), "ქ": (3, 7), "ბ": (3, 8), "ჰ": (3, 9),
}

# Devanagari InScript (the Indian national standard layout; ne types
# Devanagari. The Nepali "Traditional Romanized" layout is more common in
# Nepal but the InScript grid is the documented standard — same honesty
# rule as the other curated grids).
_NE_INSCRIPT = {
    "`": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "-": (0, 11), "ृ": (0, 12),
    "ौ": (1, 0), "ै": (1, 1), "ा": (1, 2), "ी": (1, 3), "ू": (1, 4), "ब": (1, 5),
    "ह": (1, 6), "ग": (1, 7), "द": (1, 8), "ज": (1, 9), "ड": (1, 10), "़": (1, 11),
    "ो": (2, 0), "े": (2, 1), "्": (2, 2), "ि": (2, 3), "ु": (2, 4), "प": (2, 5),
    "र": (2, 6), "क": (2, 7), "त": (2, 8), "च": (2, 9), "ट": (2, 10),
    "ॉ": (3, 0), "ं": (3, 1), "म": (3, 2), "न": (3, 3), "व": (3, 4), "ल": (3, 5),
    "स": (3, 6), ",": (3, 7), ".": (3, 8), "य": (3, 9),
}

# Croatian/Slovenian QWERTZ (KBDYCL.DLL — š đ č ć ž are real keys).
_HRSL = {
    "‚": (0, 0), "1": (0, 1), "2": (0, 2), "3": (0, 3), "4": (0, 4), "5": (0, 5),
    "6": (0, 6), "7": (0, 7), "8": (0, 8), "9": (0, 9), "0": (0, 10), "'": (0, 11), "+": (0, 12),
    "q": (1, 0), "w": (1, 1), "e": (1, 2), "r": (1, 3), "t": (1, 4), "z": (1, 5),
    "u": (1, 6), "i": (1, 7), "o": (1, 8), "p": (1, 9), "š": (1, 10), "đ": (1, 11),
    "a": (2, 0), "s": (2, 1), "d": (2, 2), "f": (2, 3), "g": (2, 4), "h": (2, 5),
    "j": (2, 6), "k": (2, 7), "l": (2, 8), "č": (2, 9), "ć": (2, 10), "ž": (2, 11),
    "y": (3, 0), "x": (3, 1), "c": (3, 2), "v": (3, 3), "b": (3, 4), "n": (3, 5),
    "m": (3, 6), ",": (3, 7), ".": (3, 8), "-": (3, 9),
}

_LAYOUTS = {
    "qwerty": _QWERTY,
    "qwertz": _QWERTZ,
    "azerty": _AZERTY,
    "jcuken": _JCUKEN,
    "trq": _TR_Q,
    "uk_jcuken": _UK_JCUKEN,
    "el_phonetic": _EL_PHONETIC,
    # Plan 83 batch 2 national grids.
    "ar_101": _AR_101,
    "fa_std": _FA_STD,
    "he_si1452": _HE_SI1452,
    "bg_bds": _BG_BDS,
    "sr_cyr": _SR_CYR,
    "mk_cyr": _MK_CYR,
    "mn_cyr": _MN_CYR,
    "hy_phonetic": _HY_PHONETIC,
    "ka_national": _KA_NATIONAL,
    "devanagari_inscript": _NE_INSCRIPT,
    "hrsl": _HRSL,
}

LANG_LAYOUT = {
    "en": "qwerty",
    "es": "qwerty",
    "pt": "qwerty",
    "de": "qwertz",
    "fr": "azerty",
    "ru": "jcuken",
    # Plan 77 coverage expansion: Latin-script newcomers reuse the gem's
    # qwerty/qwertz grids; tr/uk/el use the curated national grids above.
    "it": "qwerty",
    "nl": "qwerty",
    "pl": "qwerty",
    "sv": "qwerty",
    "da": "qwerty",
    "ca": "qwerty",
    "ro": "qwerty",
    "vi": "qwerty",
    "cs": "qwertz",
    "hu": "qwertz",
    "tr": "trq",
    "uk": "uk_jcuken",
    "el": "el_phonetic",
    # Plan 83 batch 2: gem-wired RTL languages first (ar fa he), then the
    # national-script and Latin newcomers. Latin-script newcomers reuse the
    # gem qwerty/qwertz grids with per-language diacritic supplements,
    # exactly the wave-1 pattern.
    "ar": "ar_101",
    "fa": "fa_std",
    "he": "he_si1452",
    "bg": "bg_bds",
    "sr": "sr_cyr",
    "mk": "mk_cyr",
    "mn": "mn_cyr",
    "hy": "hy_phonetic",
    "ka": "ka_national",
    "ne": "devanagari_inscript",
    "hr": "hrsl",
    "sl": "hrsl",
    "id": "qwerty",
    "sk": "qwerty",
    "et": "qwerty",
    "lt": "qwerty",
    "lv": "qwerty",
    "is": "qwerty",
    "nn": "qwerty",
    "cy": "qwerty",
    "ga": "qwerty",
    "eu": "qwerty",
    "eo": "qwerty",
    "la": "qwerty",
    "gl": "qwerty",
    "lb": "qwertz",
    "fy": "qwerty",
    "br": "qwerty",
    "gd": "qwerty",
    "oc": "qwerty",
    "ia": "qwerty",
    "nds": "qwertz",
    "tk": "qwerty",
}

# Dead-key / diacritic slips per language. Keys of the inner dict are base
# letters (which have physical keys), values are the accented variants the
# typist adds or drops. German umlauts are real keys on the gem's QWERTZ
# and therefore also have physical neighbors; they are listed here so the
# a<->ä, o<->ö, u<->ü, s<->ß spelling slips exist too.
_LANG_ALTERNATES = {
    "en": {},
    "de": {"a": ["ä"], "o": ["ö"], "u": ["ü"], "s": ["ß"]},
    "es": {"a": ["á"], "e": ["é"], "i": ["í"], "o": ["ó"], "u": ["ú", "ü"], "n": ["ñ"]},
    "fr": {"e": ["é", "è", "ê", "ë"], "a": ["à", "â"], "u": ["û"], "i": ["î", "ï"], "o": ["ô"], "c": ["ç"]},
    "pt": {"a": ["á", "â", "ã"], "e": ["é", "ê"], "i": ["í"], "o": ["ó", "ô", "õ"], "u": ["ú", "ü"], "c": ["ç"]},
    "ru": {"е": ["ё"]},
    # Plan 77 coverage expansion. Accented letters typed via dead keys or
    # AltGr on the underlying qwerty/qwertz physical layout: they inherit
    # the base letter's neighbors plus the dropped-diacritic slip.
    "it": {"a": ["à"], "e": ["è", "é"], "i": ["ì"], "o": ["ò"], "u": ["ù"]},
    "nl": {"e": ["é"], "i": ["ï"], "o": ["ö"], "u": ["ü"]},
    "pl": {"a": ["ą"], "c": ["ć"], "e": ["ę"], "l": ["ł"], "n": ["ń"], "o": ["ó"], "s": ["ś"], "z": ["ż", "ź"]},
    "cs": {"a": ["á"], "e": ["ě", "é"], "i": ["í"], "o": ["ó"], "u": ["ú", "ů"], "y": ["ý"], "c": ["č"], "s": ["š"], "r": ["ř"], "z": ["ž"], "d": ["ď"], "t": ["ť"], "n": ["ň"]},
    "hu": {"a": ["á"], "e": ["é"], "i": ["í"], "o": ["ó", "ö", "ő"], "u": ["ú", "ü", "ű"]},
    "ro": {"a": ["ă", "â"], "i": ["î"], "s": ["ș"], "t": ["ț"]},
    "sv": {"a": ["å", "ä"], "o": ["ö"], "e": ["é"]},
    "da": {"a": ["å", "æ"], "o": ["ø"], "e": ["æ", "é"]},
    "ca": {"a": ["à"], "e": ["é"], "i": ["í"], "o": ["ó"], "u": ["ú"], "c": ["ç"]},
    "vi": {
        "a": ["ă", "â", "à", "á", "ả", "ã", "ạ"],
        "e": ["ê", "è", "é", "ẻ", "ẽ", "ẹ"],
        "i": ["ì", "í", "ỉ", "ĩ", "ị"],
        "o": ["ô", "ơ", "ò", "ó", "ỏ", "õ", "ọ"],
        "u": ["ư", "ù", "ú", "ủ", "ũ", "ụ"],
        "y": ["ỳ", "ý", "ỷ", "ỹ", "ỵ"],
        "d": ["đ"],
    },
    # tr/uk/el letters are real keys on their curated national grids, so
    # only the cross-script spelling slips get supplement entries.
    "tr": {"i": ["ı"], "o": ["ô"], "u": ["û"]},
    "uk": {},
    "el": {
        "α": ["ά"], "ε": ["έ"], "η": ["ή"], "ι": ["ί"], "ο": ["ό"], "υ": ["ύ"], "ω": ["ώ"],
        "σ": ["ς"],
    },
    # Plan 83 batch 2 supplements.
    # RTL grids: hamza carriers and Arabic-script spelling slips. The
    # Persian keheh/yeh forms are the classic Persian typos (Arabic kaf/yeh
    # instead of the Persian letters); alef madda/hamza-below for Arabic.
    "ar": {"ا": ["أ", "إ", "آ"], "ي": ["ى"], "ه": ["ة"]},
    "fa": {"ی": ["ي"], "ک": ["ك"], "ا": ["آ", "أ"]},
    "he": {},
    # Cyrillic national grids carry all letters as real keys.
    "bg": {}, "sr": {}, "mk": {"ђ": ["ѓ"], "ћ": ["ќ"]}, "mn": {"у": ["ү"], "о": ["ө"]},
    "hy": {}, "ka": {},
    # Devanagari: dental/retroflex nasal slip.
    "ne": {"न": ["ण"]},
    # hr/sl letters are real keys on the shared QWERTZ grid.
    "hr": {}, "sl": {},
    # Latin newcomers: dead-key/AltGr diacritics over qwerty/qwertz.
    "id": {},
    "sk": {"a": ["á", "ä"], "e": ["é"], "i": ["í"], "o": ["ó", "ô"], "u": ["ú"],
           "y": ["ý"], "l": ["ĺ", "ľ"], "r": ["ŕ"], "c": ["č"], "s": ["š"],
           "n": ["ň"], "z": ["ž"], "d": ["ď"], "t": ["ť"]},
    "et": {"a": ["ä"], "o": ["ö", "õ"], "u": ["ü"], "s": ["š"], "z": ["ž"]},
    "lt": {"a": ["ą"], "c": ["č"], "e": ["ę", "ė"], "i": ["į"], "s": ["š"],
           "u": ["ų", "ū"], "z": ["ž"]},
    "lv": {"a": ["ā"], "c": ["č"], "e": ["ē"], "g": ["ģ"], "i": ["ī"], "k": ["ķ"],
           "l": ["ļ"], "n": ["ņ"], "s": ["š"], "u": ["ū"], "z": ["ž"]},
    "is": {"a": ["á", "æ"], "e": ["é"], "i": ["í"], "o": ["ó", "ö"], "u": ["ú"],
           "y": ["ý"], "t": ["þ"], "d": ["ð"]},
    "nn": {"o": ["ø"], "a": ["å", "æ"], "e": ["é", "è"]},
    "cy": {"a": ["â", "á"], "e": ["ê", "é"], "i": ["î"], "o": ["ô", "ó"],
           "u": ["û", "ú"], "w": ["ŵ"], "y": ["ŷ"]},
    "ga": {"a": ["á"], "e": ["é"], "i": ["í"], "o": ["ó"], "u": ["ú"]},
    "eu": {"n": ["ñ"], "c": ["ç"]},
    "eo": {"c": ["ĉ"], "g": ["ĝ"], "h": ["ĥ"], "j": ["ĵ"], "s": ["ŝ"], "u": ["ŭ"]},
    "la": {"a": ["ā"], "e": ["ē"], "i": ["ī"], "o": ["ō"], "u": ["ū"]},
    "gl": {"a": ["á", "à"], "e": ["é", "è"], "i": ["í"], "o": ["ó", "ò"],
           "u": ["ú"], "n": ["ñ"], "c": ["ç"]},
    "lb": {"a": ["ä"], "e": ["é", "ë"]},
    "fy": {"a": ["â"], "e": ["ê", "é"], "i": ["î"], "o": ["ô"], "u": ["û", "ú"]},
    "br": {"a": ["â"], "e": ["ê"], "i": ["î"], "o": ["ô"], "u": ["û"], "n": ["ñ"]},
    "gd": {"a": ["à"], "e": ["è"], "i": ["ì"], "o": ["ò"], "u": ["ù"]},
    "oc": {"a": ["á", "à"], "e": ["é", "è"], "i": ["í", "ï"], "o": ["ó", "ò", "ö"],
           "u": ["ú", "ù"], "n": ["ñ"], "c": ["ç"]},
    "ia": {"a": ["á"], "e": ["é"], "i": ["í"], "o": ["ó"], "u": ["ú"]},
    "nds": {"a": ["ä"], "o": ["ö"], "u": ["ü"]},
    "tk": {"a": ["ä"], "c": ["ç"], "j": ["ž"], "n": ["ň"], "o": ["ö"],
           "s": ["ş"], "u": ["ü"], "y": ["ý"]},
}

# ---------------------------------------------------------------------------
# CJK confusion sets — curated, not learned.
# ---------------------------------------------------------------------------

# Japanese: kanji lookalikes + kana lookalikes (incl. kanji/katakana
# homoglyphs) from well-known Japanese proofreading confusion lists.
_JA_PAIRS = [
    ("未", "末"), ("士", "土"), ("汁", "汗"), ("人", "入"), ("大", "太"), ("犬", "太"),
    ("天", "夭"), ("玉", "王"), ("刀", "刃"), ("力", "刀"), ("干", "千"), ("午", "牛"),
    ("目", "日"), ("曰", "日"), ("白", "自"), ("田", "由"), ("由", "甲"), ("甲", "申"),
    ("治", "冶"), ("幸", "辛"), ("免", "兔"), ("微", "徴"), ("縁", "緑"), ("侯", "候"),
    ("糸", "系"), ("万", "方"),
    ("い", "り"), ("ね", "れ"), ("る", "ろ"), ("わ", "れ"), ("め", "ぬ"), ("は", "ほ"),
    ("シ", "ツ"), ("ソ", "ン"), ("ハ", "ニ"), ("ー", "一"),
    ("口", "ロ"), ("工", "エ"), ("力", "カ"), ("二", "ニ"), ("八", "ハ"), ("三", "ミ"),
]

# Korean jamo pairs: vowel mergers (identical modern pronunciation), y-vowel
# slips, shape flips, and fortis/lenis/aspirated consonant confusion.
# Applied to the decomposed jamo inside Hangul syllables.
_KO_PAIRS = [
    ("ㅐ", "ㅔ"), ("ㅒ", "ㅖ"), ("ㅙ", "ㅞ"), ("ㅚ", "ㅙ"), ("ㅚ", "ㅞ"), ("ㅟ", "ㅢ"),
    ("ㅘ", "ㅝ"),
    ("ㅏ", "ㅓ"), ("ㅑ", "ㅕ"), ("ㅗ", "ㅜ"), ("ㅛ", "ㅠ"),
    ("ㅏ", "ㅑ"), ("ㅓ", "ㅕ"), ("ㅗ", "ㅛ"), ("ㅜ", "ㅠ"),
    ("ㄱ", "ㅋ"), ("ㄷ", "ㅌ"), ("ㅂ", "ㅍ"), ("ㅈ", "ㅊ"), ("ㅅ", "ㅆ"),
]

# Simplified Chinese shape-similar hanzi pairs.
_ZH_PAIRS = [
    ("己", "已"), ("已", "巳"), ("日", "曰"), ("土", "士"), ("未", "末"), ("太", "大"),
    ("大", "犬"), ("王", "玉"), ("木", "本"), ("天", "夭"), ("千", "干"), ("午", "牛"),
    ("入", "人"), ("刀", "力"), ("儿", "几"), ("目", "自"), ("水", "永"), ("鸟", "乌"),
    ("免", "兔"), ("斤", "斥"), ("戊", "戌"), ("戌", "戍"), ("乞", "气"), ("爪", "瓜"),
    ("今", "令"), ("云", "去"), ("拔", "拨"), ("微", "徴"), ("侯", "候"),
]

_LEADS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_VOWELS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_TAILS = "\0ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"  # \0 = no final


def _pair_map(pairs):
    m: dict[str, list[str]] = {}
    for a, b in pairs:
        m.setdefault(a, [])
        if b not in m[a]:
            m[a].append(b)
        m.setdefault(b, [])
        if a not in m[b]:
            m[b].append(a)
    return m


_CONFUSION = {
    "ja": _pair_map(_JA_PAIRS),
    "zh": _pair_map(_ZH_PAIRS),
}
_CONFUSION_KO = _pair_map(_KO_PAIRS)

CJK_LANGS = ("ja", "ko", "zh")


def _hangul_decompose(ch: str):
    code = ord(ch) - 0xAC00
    if not 0 <= code < 11172:
        return None
    return _LEADS[code // 588], _VOWELS[(code % 588) // 28], _TAILS[code % 28]


def _hangul_compose(lead: str, vowel: str, tail: str) -> str:
    return chr(0xAC00 + (_LEADS.index(lead) * 21 + _VOWELS.index(vowel)) * 28 + _TAILS.index(tail))


# ---------------------------------------------------------------------------
# Neighbor tables for the layout languages (built once at import).
# ---------------------------------------------------------------------------


def _build_neighbor_table(layout: dict, alternates: dict) -> dict:
    """char -> (np.ndarray candidate chars, np.ndarray probabilities)."""
    table: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ch, pos in layout.items():
        cands: list[tuple[str, float]] = []
        for other, opos in layout.items():
            if other == ch:
                continue
            d = abs(pos[0] - opos[0]) + abs(pos[1] - opos[1])
            if d == 1:
                cands.append((other, WEIGHT_D1))
            elif d == 2:
                cands.append((other, WEIGHT_D2))
        for variant in alternates.get(ch, ()):
            cands.append((variant, WEIGHT_ALTERNATE))
        if cands:
            table[ch] = _weighted(cands)

    # Accented letters have no key of their own: they inherit the base
    # letter's physical neighbors, plus the base letter itself (dropped
    # diacritic) and their diacritic siblings.
    for base, variants in alternates.items():
        physical = []
        for other, opos in layout.items():
            if other == base:
                continue
            bpos = layout[base]
            d = abs(bpos[0] - opos[0]) + abs(bpos[1] - opos[1])
            if d == 1:
                physical.append((other, WEIGHT_D1))
            elif d == 2:
                physical.append((other, WEIGHT_D2))
        for v in variants:
            cands = list(physical)
            cands.append((base, WEIGHT_BASE))
            for sibling in variants:
                if sibling != v:
                    cands.append((sibling, WEIGHT_ALTERNATE))
            table[v] = _weighted(cands)
    return table


def _weighted(cands: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    chars = np.array([c for c, _ in cands], dtype="<U2")
    w = np.array([w for _, w in cands], dtype=np.float64)
    return chars, w / w.sum()


_TABLES = {
    lang: _build_neighbor_table(_LAYOUTS[name], _LANG_ALTERNATES[lang])
    for lang, name in LANG_LAYOUT.items()
}


# ---------------------------------------------------------------------------
# Operation primitives (rng-only randomness).
# ---------------------------------------------------------------------------


def _draw_neighbor(rng: np.random.Generator, table: dict, ch: str) -> str | None:
    entry = table.get(ch)
    if entry is None:
        return None
    chars, probs = entry
    return str(chars[int(rng.choice(chars.size, p=probs))])


def _op_substitute(rng: np.random.Generator, word: str, table: dict) -> str | None:
    positions = [i for i, ch in enumerate(word) if ch in table]
    if not positions:
        return None
    i = positions[int(rng.integers(len(positions)))]
    c = _draw_neighbor(rng, table, word[i])
    if c is None or c == word[i]:
        return None
    return word[:i] + c + word[i + 1 :]


def _op_insert(rng: np.random.Generator, word: str, table: dict) -> str | None:
    pos = int(rng.integers(len(word) + 1))
    anchor = word[pos] if pos < len(word) else word[pos - 1]
    c = _draw_neighbor(rng, table, anchor)
    if c is None:
        return None
    return word[:pos] + c + word[pos:]


def _op_delete(rng: np.random.Generator, word: str) -> str | None:
    if len(word) < 2:
        return None
    i = int(rng.integers(len(word)))
    return word[:i] + word[i + 1 :]


def _op_transpose(rng: np.random.Generator, word: str) -> str | None:
    if len(word) < 2:
        return None
    i = int(rng.integers(len(word) - 1))
    if word[i] == word[i + 1]:
        return None
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def _op_confuse(rng: np.random.Generator, word: str, lang: str) -> str | None:
    """Substitute one character with a curated confusion partner."""
    if lang == "ko":
        return _op_confuse_ko(rng, word)
    confusion = _CONFUSION[lang]
    positions = [i for i, ch in enumerate(word) if ch in confusion]
    if not positions:
        return None
    i = positions[int(rng.integers(len(positions)))]
    partners = confusion[word[i]]
    c = partners[int(rng.integers(len(partners)))]
    return word[:i] + c + word[i + 1 :]


def _op_confuse_ko(rng: np.random.Generator, word: str) -> str | None:
    """Substitute one confusable jamo inside a Hangul syllable."""
    slots = []  # (word index, "lead"/"vowel"/"tail", jamo)
    for i, ch in enumerate(word):
        parts = _hangul_decompose(ch)
        if parts is None:
            continue
        lead, vowel, tail = parts
        if lead in _CONFUSION_KO:
            slots.append((i, "lead", lead))
        if vowel in _CONFUSION_KO:
            slots.append((i, "vowel", vowel))
        if tail != "\0" and tail in _CONFUSION_KO:
            slots.append((i, "tail", tail))
    if not slots:
        return None
    i, which, jamo = slots[int(rng.integers(len(slots)))]
    partners = _CONFUSION_KO[jamo]
    partner = partners[int(rng.integers(len(partners)))]
    lead, vowel, tail = _hangul_decompose(word[i])
    if which == "lead":
        lead = partner
    elif which == "vowel":
        vowel = partner
    else:
        tail = partner
    new = _hangul_compose(lead, vowel, tail)
    if new == word[i]:
        return None
    return word[:i] + new + word[i + 1 :]


def _pick(rng: np.random.Generator, weighted_ops: tuple) -> object:
    u = rng.random()
    acc = 0.0
    for name, w in weighted_ops:
        acc += w
        if u < acc:
            return name
    return weighted_ops[-1][0]


def make_typo(rng: np.random.Generator, word: str, lang: str) -> str | None:
    """Return one deterministic corrupted variant of ``word``.

    Tries up to TYPO_ATTEMPTS single-edit corruptions and returns the first
    that changes the word; returns None if the word resists corruption
    (e.g. no mappable character and nothing to delete/transpose). The
    caller is responsible for vocabulary membership, as in run_eval.
    """
    if not word:
        return None

    if lang in CJK_LANGS:
        confusable = _has_confusable(word, lang)
        ops = OP_WEIGHTS_CJK if confusable else (("transpose", 1.0),)
        for _ in range(TYPO_ATTEMPTS):
            op = _pick(rng, ops)
            typo = _op_confuse(rng, word, lang) if op == "substitute" else _op_transpose(rng, word)
            if typo is not None and typo != word:
                return typo
        return None

    table = _TABLES[lang]
    ops: list = [OP_WEIGHTS_LAYOUT[0], OP_WEIGHTS_LAYOUT[1]]  # substitute, insert
    if len(word) >= 2:
        ops.append(OP_WEIGHTS_LAYOUT[2])
        ops.append(OP_WEIGHTS_LAYOUT[3])
    total = sum(w for _, w in ops)
    ops = tuple((name, w / total) for name, w in ops)
    for _ in range(TYPO_ATTEMPTS):
        op = _pick(rng, ops)
        if op == "substitute":
            typo = _op_substitute(rng, word, table)
        elif op == "insert":
            typo = _op_insert(rng, word, table)
        elif op == "delete":
            typo = _op_delete(rng, word)
        else:
            typo = _op_transpose(rng, word)
        if typo is not None and typo != word:
            return typo
    return None


def _has_confusable(word: str, lang: str) -> bool:
    if lang == "ko":
        for ch in word:
            parts = _hangul_decompose(ch)
            if parts is None:
                continue
            lead, vowel, tail = parts
            if lead in _CONFUSION_KO or vowel in _CONFUSION_KO or (tail != "\0" and tail in _CONFUSION_KO):
                return True
        return False
    confusion = _CONFUSION[lang]
    return any(ch in confusion for ch in word)


def main() -> int:
    """Smoke test: deterministic typos per language on sample words."""
    import zlib

    samples = {
        "en": ["hello", "world", "keyboard"],
        "de": ["schön", "straße", "übermorgen"],
        "es": ["señor", "corazón", "español"],
        "fr": ["réponse", "français", "déjà"],
        "pt": ["coração", "ação", "português"],
        "ru": ["привет", "ещё", "молоко"],
        "ja": ["今日は", "先生", "日本語"],
        "ko": ["안녕하세요", "개발자", "학교"],
        "zh": ["你好", "我们", "电脑"],
        "it": ["città", "perché", "morire"],
        "nl": ["volgende", "moeilijk", "zachte"],
        "pl": ["szkoła", "wszystkie", "ćwiczenie"],
        "uk": ["привіт", "Україна", "будь"],
        "tr": ["ışık", "ağaç", "gözlük"],
        "cs": ["člověk", "velký", "malíř"],
        "sv": ["träna", "svenska", "första"],
        "el": ["καλημέρα", "καληνύχτα", "ερώτηση"],
        "hu": ["magyar", "körül", "kicsi"],
        "ro": ["română", "mâine", "școală"],
        "da": ["hvordan", "små", "søge"],
        "vi": ["việt", "người", "học"],
        "ca": ["català", "cert", "més"],
        # Plan 83 batch 2.
        "ar": ["كتاب", "مدرسة", "عربية"],
        "fa": ["کتاب", "مدرسه", "فارسی"],
        "he": ["שלום", "עברית", "בית"],
        "bg": ["книга", "училище", "български"],
        "sr": ["књига", "школа", "српски"],
        "mk": ["книга", "училиште", "македонски"],
        "mn": ["ном", "сургууль", "монгол"],
        "hy": ["գիրք", "դպրոց", "հայերեն"],
        "ka": ["წიგნი", "სკოლა", "ქართული"],
        "ne": ["किताब", "विद्यालय", "नेपाली"],
        "hr": ["knjiga", "škola", "hrvatski"],
        "sl": ["knjiga", "šola", "slovenščina"],
        "id": ["buku", "sekolah", "indonesia"],
        "sk": ["kniha", "škola", "slovenčina"],
        "et": ["raamat", "kool", "eesti"],
        "lt": ["knyga", "mokykla", "lietuvių"],
        "lv": ["grāmata", "skola", "latviešu"],
        "is": ["bók", "skóli", "íslenska"],
        "nn": ["bok", "skule", "norsk"],
        "cy": ["llyfr", "ysgol", "cymraeg"],
        "ga": ["leabhar", "scoil", "gaeilge"],
        "eu": ["liburu", "eskola", "euskara"],
        "eo": ["libro", "lernejo", "esperanto"],
        "la": ["liber", "schola", "latina"],
        "gl": ["libro", "escola", "galego"],
        "lb": ["buch", "schoul", "lëtzebuergesch"],
        "fy": ["boek", "skoalle", "frysk"],
        "br": ["levr", "skol", "brezhoneg"],
        "gd": ["leabhar", "sgoil", "gàidhlig"],
        "oc": ["libre", "escòla", "occitan"],
        "ia": ["libro", "schola", "interlingua"],
        "nds": ["book", "school", "plattdüütsch"],
        "tk": ["kitap", "mekdep", "türkmen"],
    }
    for lang, words in samples.items():
        rng = np.random.default_rng([42, zlib.crc32(lang.encode("utf-8"))])
        out = []
        for w in words:
            out.append(f"{w} -> {make_typo(rng, w, lang)}")
        print(f"{lang}: " + "  ".join(out))
    # determinism self-check
    rng1 = np.random.default_rng([42, zlib.crc32("de".encode("utf-8"))])
    rng2 = np.random.default_rng([42, zlib.crc32("de".encode("utf-8"))])
    word = "schönes"
    a = [make_typo(rng1, word, "de") for _ in range(20)]
    b = [make_typo(rng2, word, "de") for _ in range(20)]
    assert a == b, "non-deterministic typo stream"
    print("determinism self-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
