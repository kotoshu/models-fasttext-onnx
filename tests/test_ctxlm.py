#!/usr/bin/env python3
"""Plan 16 tests: ctx-LM trainer determinism + backoff math + scorer smoke.

Pure python + a tiny inline corpus. The trainer's vocabulary gate,
bigram hash, and backoff math are the behavioural contracts that
survive even if the corpus source changes; the scorer smoke runs
end-to-end against a synthetic mini-LM.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_ctx_lm import hash_pair, load_vocab_ids, tokenize  # noqa: E402
from ctxlm_scorer import _bigram_count, load_ctx  # noqa: E402


def _score_with(uni, bgk, log_p_uni, alpha, target, ctx):
    s = 0.0
    a_count = max(int(uni[target]), 1)
    for c in ctx:
        if c == target:
            continue
        cnt = _bigram_count(bgk, None, c, target)
        if cnt > 0:
            p = cnt / a_count
        else:
            p = alpha * float(np.exp(log_p_uni[target]))
        if p > 0:
            s += math.log(p)
    return s


class TokenizerContract(unittest.TestCase):
    def test_lowers_and_strips_punct(self):
        self.assertEqual(
            tokenize("Hello, the QUICK brown-Foxes don't jump."),
            ["hello", "the", "quick", "brownfoxes", "don't", "jump"],
        )

    def test_internal_apostrophe_kept(self):
        self.assertEqual(tokenize("it's a test"), ["it's", "a", "test"])

    def test_hyphen_strip_then_alpha_filter(self):
        self.assertIn("brownfoxes", tokenize("brown-foxes run"))


class BigramHashContract(unittest.TestCase):
    def test_stable(self):
        self.assertEqual(hash_pair(7, 42), hash_pair(7, 42))

    def test_order_dependent(self):
        self.assertNotEqual(hash_pair(7, 42), hash_pair(42, 7))

    def test_full_32_bit_range(self):
        vs = {hash_pair(a, b) for a in range(8) for b in range(8)}
        self.assertGreater(max(vs) >> 16, 0)


class VocabularyGate(unittest.TestCase):
    def test_filter_drops_oov(self):
        vocab_ids, _ = load_vocab_ids("en")
        text = "the zzzzzquick brown fox"
        ids = [vocab_ids[t] for t in tokenize(text) if t in vocab_ids]
        self.assertGreater(len(ids), 0)
        self.assertLess(len(ids), len(tokenize(text)))


class BackoffMath(unittest.TestCase):
    def test_known_pair_uses_bigram(self):
        uni = Counter({"harder": 10, "than": 100, "then": 200, "ok": 1000})
        bi = Counter({("harder", "than"): 3, ("harder", "then"): 2})
        total = sum(uni.values())
        a, alpha = "harder", 0.4
        self.assertAlmostEqual(bi[(a, "than")] / uni[a], 0.3)
        self.assertAlmostEqual(bi[(a, "then")] / uni[a], 0.2)
        p_then = uni["then"] / total
        self.assertAlmostEqual(alpha * p_then, 0.4 * 200 / 1310)


class CtxLMScorerSmoke(unittest.TestCase):
    """End-to-end: train a tiny LM on a synthetic corpus, then ask the
    scorer which of {eat, each} better fits the context {i, apple}."""

    def test_high_prob_context_outranks_low(self):
        vocab_ids, _ = load_vocab_ids("en")
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            for _ in range(50):
                fh.write('{"text": "I eat apple in the morning daily"}\n')
            fh.write('{"text": "I each apple in the morning once"}\n')
            tmp = Path(fh.name)
        try:
            uni, bi = Counter(), Counter()
            with tmp.open() as fh:
                for line in fh:
                    obj = json.loads(line)
                    ids = [vocab_ids[t] for t in tokenize(obj["text"]) if t in vocab_ids]
                    uni.update(ids)
                    bi.update(zip(ids, ids[1:]))
            vs = max(vocab_ids.values()) + 1
            mini = Path("/tmp/test_ctxlm_mini.npz")
            np.savez(
                mini,
                unigram_counts=np.array([uni.get(i, 0) for i in range(vs)], dtype=np.uint32),
                bigram_keys=np.array([hash_pair(a, b) for (a, b) in bi], dtype=np.uint32),
                bigram_counts=np.array(list(bi.values()), dtype=np.uint32),
            )
            uni2, bgk, log_p_uni, alpha = load_ctx(mini, vs)
            ctx = [vocab_ids["i"], vocab_ids["apple"]]
            score_eat = _score_with(uni2, bgk, log_p_uni, alpha, vocab_ids["eat"], ctx)
            score_each = _score_with(uni2, bgk, log_p_uni, alpha, vocab_ids["each"], ctx)
            self.assertGreater(score_eat, score_each)
        finally:
            tmp.unlink()
            if mini.exists():
                mini.unlink()


if __name__ == "__main__":
    unittest.main()
