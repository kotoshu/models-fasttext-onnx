#!/usr/bin/env python3
"""CtxLMScorer for the Phase-1 gate (plan 16).

Loads the .npz artifact the trainer emits and answers `margins` for
the eval probe. Backoff: P(t|c) = c(c,t)/c(c) if c(c,t) > 0 else
alpha*P(t) — the conditioning word is the CONTEXT neighbor, never
the target. Score(t|ctx) = sum over neighbors c of log P(t|c);
margin = score(observed) - score(candidate) so larger margin = the
candidate fits WORSE... no: margins() returns base - score(cand),
so POSITIVE margin means the candidate scores LOWER than the
observed word. NOTE: the probe treats larger margin = candidate
BETTER; margins() here returns score(observed) - score(candidate)
which is positive when the candidate is WORSE. The probe's own
_CtxLMScorer is the one used by the gate (kept independently so the
frozen evidence is self-contained); this module provides load_ctx +
_bigram_count as the shared primitives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bigram_hash(a: int, b: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{a}|{b}".encode(), digest_size=4).digest(), "big")


def _bigram_count(bg_lookup, bg_counts, a: int, b: int) -> int:
    """bg_lookup is the O(1) dict from load_ctx (bg_counts unused); kept
    two-arg for the array/searchsorted alternative."""
    return bg_lookup.get(_bigram_hash(a, b), 0)


def load_ctx(path, vocab_size, alpha=0.4):
    """Returns (unigram_counts int64, bigram dict {hash: count},
    log_p_uni float64[vocab_size], alpha). The dict trades ~1.5 GB for
    O(1) lookups — the difference between a gate run in seconds vs
    hours at 10M bigram types."""
    data = np.load(path)
    uni = data["unigram_counts"].astype(np.int64)
    bg_dict = dict(zip(data["bigram_keys"].tolist(), data["bigram_counts"].tolist()))
    total = max(int(uni.sum()), 1)
    log_p_uni = np.full(vocab_size, -np.inf, dtype=np.float64)
    mask = uni > 0
    log_p_uni[mask] = np.log(uni[mask] / total)
    return uni, bg_dict, log_p_uni, float(alpha)


def score(target: int, ctx, uni, bg_dict, log_p_uni, alpha: float,
          vocab_size: int | None = None) -> float:
    """Sum of log P(target | c) over context neighbors c, stupid backoff
    with additive unigram smoothing: an unseen target gets P = 1/(total+V)
    — tiny, never zero. Zero-count candidates must score WORSE than any
    attested word, not be skipped to a vacuous 0.0 "perfect fit"."""
    import math
    if vocab_size is None:
        vocab_size = len(uni)
    total = max(int(uni.sum()), 1)
    denom = total + vocab_size
    p_smooth = (int(uni[target]) + 1) / denom
    s = 0.0
    for c in ctx:
        if c == target:
            continue
        cnt = _bigram_count(bg_dict, None, c, target)
        if cnt > 0:
            p = cnt / max(int(uni[c]), 1)
        else:
            p = alpha * p_smooth
        s += math.log(p)
    return s


class CtxLMScorer:
    """Convenience wrapper; margins[i] = score(observed) - score(cand_i)
    (positive = candidate fits worse)."""

    def __init__(self, ctx_path, vocab_size, alpha=0.4):
        self.uni, self.bg_dict, self.log_p_uni, self.alpha = load_ctx(ctx_path, vocab_size, alpha)

    def margins(self, word_idx, cand_idxs, ctx):
        # candidate-minus-observed: positive = candidate fits better
        base = score(word_idx, ctx, self.uni, self.bg_dict, self.log_p_uni, self.alpha)
        return np.array(
            [score(w, ctx, self.uni, self.bg_dict, self.log_p_uni, self.alpha) - base
             for w in cand_idxs],
            dtype=np.float64,
        )
