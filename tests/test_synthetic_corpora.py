#!/usr/bin/env python3
"""Plan 123: the synthetic typo corpora carry their admission contract.

Structural invariants run everywhere; the dictionary-grounded checks
(correction valid / typo invalid) and the determinism rebuild run when
the dictionaries checkout is present, because the models repository
does not vendor it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import os

REPO = Path(__file__).resolve().parent.parent
CORPORA = REPO / "eval" / "corpora" / "synth"
DICT_ROOT = Path(os.environ.get("KOTOSHU_DICTIONARIES_ROOT", REPO.parent / "dictionaries"))

sys.path.insert(0, str(REPO / "eval"))
from synthesize_corpora import (  # noqa: E402
    MAX_WORD_LEN,
    MIN_WORD_LEN,
    _is_pure_letters,
    _levenshtein,
    load_dictionary_words,
)


def _load(lang: str) -> dict:
    return json.loads((CORPORA / f"{lang}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("lang", ["de", "es"])
def test_size_and_provenance(lang: str) -> None:
    corpus = _load(lang)
    assert corpus["n_pairs_unique"] >= 5000
    assert len(corpus["pairs"]) == corpus["n_pairs_unique"]
    meta = corpus["corpus"]
    assert meta["synthetic"] is True
    assert meta["generator"] == "eval/synthesize_corpora.py"
    assert meta["generator_seed"]
    assert len(meta["dictionary"]["sha256"]) == 64
    assert meta["vocab_cut"]["intersection_size"] >= 5000
    assert "caveat" in meta  # generator-domain honesty is part of the schema


@pytest.mark.parametrize("lang", ["de", "es"])
def test_pairs_are_unique(lang: str) -> None:
    pairs = _load(lang)["pairs"]
    keys = {(t, c) for t, c, _ in pairs}
    assert len(keys) == len(pairs)


@pytest.mark.parametrize("lang", ["de", "es"])
def test_admission_shape(lang: str) -> None:
    for typo, correction, count in _load(lang)["pairs"]:
        assert count == 1
        assert typo != correction
        assert MIN_WORD_LEN <= len(correction) <= MAX_WORD_LEN
        assert _is_pure_letters(correction)
        assert _is_pure_letters(typo)
        assert len(typo) <= MAX_WORD_LEN + 1
        d = _levenshtein(typo, correction)
        assert 1 <= d <= 2


@pytest.mark.parametrize("lang", ["de", "es"])
def test_histograms_match_pairs(lang: str) -> None:
    corpus = _load(lang)
    dists = {str(d): 0 for d in (1, 2)}
    for typo, correction, _ in corpus["pairs"]:
        dists[str(_levenshtein(typo, correction))] += 1
    assert corpus["extraction_stats"]["distance_histogram"] == dists
    assert sum(corpus["extraction_stats"]["ops_histogram"].values()) == len(corpus["pairs"])


@pytest.mark.skipif(not DICT_ROOT.exists(), reason="dictionaries checkout not present")
@pytest.mark.parametrize("lang", ["de", "es"])
def test_dictionary_admission(lang: str) -> None:
    dictionary, _ = load_dictionary_words(DICT_ROOT, lang)
    for typo, correction, _ in _load(lang)["pairs"]:
        assert correction in dictionary
        assert typo not in dictionary


@pytest.mark.skipif(not DICT_ROOT.exists(), reason="dictionaries checkout not present")
@pytest.mark.parametrize("lang", ["de", "es"])
def test_deterministic_rebuild(lang: str, tmp_path: None = None) -> None:
    from synthesize_corpora import synthesize

    rebuilt = synthesize(lang, 5000, DICT_ROOT, REPO)
    original = _load(lang)
    assert rebuilt["pairs"] == original["pairs"]
    assert rebuilt["extraction_stats"] == original["extraction_stats"]
