#!/usr/bin/env python3
"""Plan 16 tests: confusion table v2 bounded d<=2 + band rule.

Unittest convention (matches tests/test_synthetic_corpora.py). Pure
python, no fixture files — synthetic vocabs exercise the deletion
index identity-variant fix and the band-bounded d2 emission.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from build_confusion_tables import (  # noqa: E402
    build_confusions,
    core_variants,
    damerau,
    deletion_variants,
)


class IdentityVariantFix(unittest.TestCase):
    """The v1 deletion index never included the identity variant, so
    every insert/delete pair between len>=2 words (you/your,
    occured/occurred) was silently missed."""

    def test_deletion_variants_includes_identity(self):
        vs = deletion_variants("your")
        self.assertIn("your", vs)

    def test_v1_hole_pair_now_paired(self):
        t = build_confusions(["you", "your"])
        self.assertIn("your", t["you"])
        self.assertEqual(t["you"]["your"], ["dl1"])

    def test_occured_occurred_now_paired(self):
        t = build_confusions(["occured", "occurred"])
        self.assertIn("occurred", t["occured"])


class BoundedD2(unittest.TestCase):
    """Table v2 emits DL<=2 only when one side sits in the top-N band."""

    def test_d2_emitted_when_far_side_in_band(self):
        # vocab order matters: band = first 4 alpha
        t = build_confusions(
            ["the", "then", "than", "you", "your", "their", "there"],
            band=6,  # their sits at position 5, included
        )
        self.assertIn("there", t["their"])
        self.assertEqual(t["their"]["there"], ["dl2"])

    def test_d2_suppressed_when_neither_in_band(self):
        t = build_confusions(["the", "then", "than", "you", "your", "their", "there"], band=4)
        self.assertNotIn("there", t["their"])

    def test_punctuation_dropped(self):
        # non-alpha vocab entries never appear as either side
        t = build_confusions(["the", "then", "than", "you", "your", ",", ".", "</s>"], band=2)
        self.assertEqual(set(t.keys()), {"the", "then", "than", "you", "your"})


class DamerauBaseline(unittest.TestCase):
    def test_distances(self):
        for a, b, expected in [
            ("then", "than", 1),
            ("you", "your", 1),
            ("their", "there", 2),
            ("weather", "whether", 2),
            ("cat", "dog", 3),
            ("ab", "ba", 1),
            ("abc", "bac", 1),  # adjacent transposition
            ("abcd", "badc", 2),
        ]:
            with self.subTest(a=a, b=b):
                self.assertEqual(damerau(a, b), expected)


class CoreVariantsLemma(unittest.TestCase):
    def test_dl_le2_share_a_core(self):
        for a, b in [("their", "there"), ("weather", "whether"), ("abcd", "badc")]:
            with self.subTest(a=a, b=b):
                self.assertTrue(set(core_variants(a)) & set(core_variants(b)))


if __name__ == "__main__":
    unittest.main()
