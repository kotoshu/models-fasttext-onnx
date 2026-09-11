#!/usr/bin/env python3
"""Plan 123: the synthetic typo corpora carry their admission contract.

Structural invariants run everywhere; the dictionary-grounded checks
(correction valid / typo invalid) and the determinism rebuild run when
the dictionaries checkout is present, because the models repository
does not vendor it.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

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
    synthesize,
)


def _load(lang: str) -> dict:
    return json.loads((CORPORA / f"{lang}.json").read_text(encoding="utf-8"))


class SyntheticCorporaTest(unittest.TestCase):
    LANGS = ("de", "es")

    def test_size_and_provenance(self) -> None:
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                corpus = _load(lang)
                self.assertGreaterEqual(corpus["n_pairs_unique"], 5000)
                self.assertEqual(len(corpus["pairs"]), corpus["n_pairs_unique"])
                meta = corpus["corpus"]
                self.assertIs(meta["synthetic"], True)
                self.assertEqual(meta["generator"], "eval/synthesize_corpora.py")
                self.assertTrue(meta["generator_seed"])
                self.assertEqual(len(meta["dictionary"]["sha256"]), 64)
                self.assertGreaterEqual(meta["vocab_cut"]["intersection_size"], 5000)
                self.assertIn("caveat", meta)  # generator-domain honesty is part of the schema

    def test_pairs_are_unique(self) -> None:
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                pairs = _load(lang)["pairs"]
                keys = {(t, c) for t, c, _ in pairs}
                self.assertEqual(len(keys), len(pairs))

    def test_admission_shape(self) -> None:
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                for typo, correction, count in _load(lang)["pairs"]:
                    self.assertEqual(count, 1)
                    self.assertNotEqual(typo, correction)
                    self.assertTrue(MIN_WORD_LEN <= len(correction) <= MAX_WORD_LEN)
                    self.assertTrue(_is_pure_letters(correction))
                    self.assertTrue(_is_pure_letters(typo))
                    self.assertLessEqual(len(typo), MAX_WORD_LEN + 1)
                    d = _levenshtein(typo, correction)
                    self.assertTrue(1 <= d <= 2)

    def test_histograms_match_pairs(self) -> None:
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                corpus = _load(lang)
                dists = {str(d): 0 for d in (1, 2)}
                for typo, correction, _ in corpus["pairs"]:
                    dists[str(_levenshtein(typo, correction))] += 1
                self.assertEqual(corpus["extraction_stats"]["distance_histogram"], dists)
                self.assertEqual(
                    sum(corpus["extraction_stats"]["ops_histogram"].values()),
                    len(corpus["pairs"]),
                )

    @unittest.skipIf(not DICT_ROOT.exists(), "dictionaries checkout not present")
    def test_dictionary_admission(self) -> None:
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                dictionary, _ = load_dictionary_words(DICT_ROOT, lang)
                for typo, correction, _ in _load(lang)["pairs"]:
                    self.assertIn(correction, dictionary)
                    self.assertNotIn(typo, dictionary)

    @unittest.skipIf(not DICT_ROOT.exists(), "dictionaries checkout not present")
    def test_deterministic_rebuild(self) -> None:
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                rebuilt = synthesize(lang, 5000, DICT_ROOT, REPO)
                original = _load(lang)
                self.assertEqual(rebuilt["pairs"], original["pairs"])
                self.assertEqual(rebuilt["extraction_stats"], original["extraction_stats"])


if __name__ == "__main__":
    unittest.main()
