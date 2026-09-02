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

_LAYOUTS = {
    "qwerty": _QWERTY,
    "qwertz": _QWERTZ,
    "azerty": _AZERTY,
    "jcuken": _JCUKEN,
}

LANG_LAYOUT = {
    "en": "qwerty",
    "es": "qwerty",
    "pt": "qwerty",
    "de": "qwertz",
    "fr": "azerty",
    "ru": "jcuken",
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
