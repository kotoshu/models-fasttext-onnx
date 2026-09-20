#!/usr/bin/env python3
"""Calibration probe: can a margin detector flag real-word errors? (plan 15)

Two scorers on the ladder (docs/realword-detection-design.md), plus a
control:

  cosine   v0, the shipped context math (rs CosineReranker / gem
           rank_by_context): score(w|ctx) = sum of cosines between w's
           tier vector and each context neighbor's tier vector.
  skipgram fastText's own conditional scoring: score(w|ctx) =
           out(w) . sum(in_raw(c) for c in ctx), with output rows
           extracted from the training .bin
           (scripts/extract_output_matrix.py). NOTE: the shipped
           cc.*.300 output matrices are zeroed — kept as the documented
           dead rung; the run produces the all-zero evidence.
  freq     context-free control (vocab rank gap) — the floor any
           context scorer must beat.

A real word w is flagged when some confusion candidate x scores higher
by more than the margin threshold tau: score(x) - score(w) > tau.

Measured on BOTH sides, because a real-word detector lives or dies on
precision:
  recall  — eval/realword/{lang}.json pair instances (typo in-vocab,
            correction in-vocab, real sentence, diff index): flagged at
            tau, and flagged with the TRUE correction as the argmax
            candidate (the product metric: flag + right suggestion).
  FP rate — clean sentences ({lang}.clean.jsonl, the corrected sides):
            fraction of eligible in-vocab tokens flagged at tau.

Operating points are anchored on clean-margin quantiles (tau chosen so
the FP rate IS 0.5/1/2/5/10%), plus a fixed grid for comparability.
Thresholds are FROZEN from this evidence, never tuned in an engine.

Usage:
  python scripts/eval_realword_detection.py --lang en [--scorer skipgram]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_corpus import _clean_token  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TAUS = [round(t, 3) for t in np.arange(0.00, 0.31, 0.02)]
FP_ANCHORS = [0.995, 0.99, 0.98, 0.95, 0.90]  # clean-margin quantiles -> operating points
WINDOW = 5  # context tokens each side, the reranker's operating scale
MIN_CONTEXT_SUPPORT = 100  # min unigram count for a context word's MLE to be trusted


def load_tier(lang: str) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Vocab, raw tier vectors, L2-normalized tier vectors."""
    model = onnx.load(REPO_ROOT / f"models/{lang}/fasttext.{lang}.onnx")
    raw = numpy_helper.to_array(model.graph.node[0].attribute[0].t).astype(np.float32)
    vocab = json.loads((REPO_ROOT / f"models/{lang}/fasttext.{lang}.vocab.json").read_text())["word_to_idx"]
    return vocab, raw, raw / np.linalg.norm(raw, axis=1, keepdims=True)


class CosineScorer:
    """score(w|ctx) = sum of cos(w, c) over context neighbors."""

    def __init__(self, _, normalized: np.ndarray):
        self.vectors = normalized

    def margins(self, word_idx: int, cand_idxs: list[int], ctx: list[int]) -> np.ndarray:
        sel = np.array([word_idx] + cand_idxs)
        scores = (self.vectors[sel] @ self.vectors[ctx].T).sum(axis=1)
        return scores[1:] - scores[0]


class SkipgramScorer:
    """score(w|ctx) = out(w) . sum(in_raw(c)) — fastText skipgram scoring."""

    def __init__(self, output: np.ndarray, raw: np.ndarray):
        self.output, self.raw = output, raw

    def margins(self, word_idx: int, cand_idxs: list[int], ctx: list[int]) -> np.ndarray:
        ctx_vec = self.raw[ctx].sum(axis=0)
        sel = np.array([word_idx] + cand_idxs)
        scores = self.output[sel] @ ctx_vec
        return scores[1:] - scores[0]


class FreqScorer:
    """Context-free control: margin = vocab rank gap (positive = candidate
    more frequent). The floor any context scorer must beat — misspellings
    are rarer than their corrections, so rank alone 'detects'."""

    def __init__(self, _, __):
        pass

    def margins(self, word_idx: int, cand_idxs: list[int], ctx: list[int]) -> np.ndarray:
        return np.array([word_idx - c for c in cand_idxs], dtype=np.float64)


SCORERS = {"cosine": CosineScorer, "skipgram": SkipgramScorer, "freq": FreqScorer}

NEURAL_WINDOW = 8  # matches scripts/modal_train_ctx_neural.py


class _NeuralScorer:
    """The plan-17 cloze transformer: S(c, ctx) = log P(c | +/-8 window,
    gap-position encoding). Ordered context is required, so the harness
    passes the sentence and target index through `margins_ctx`."""

    def __init__(self, onnx_path, vocab):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.vocab = vocab
        self.pad = max(vocab.values()) + 1
        ctx_len = 2 * NEURAL_WINDOW
        self.pos = np.concatenate([np.arange(NEURAL_WINDOW),
                                   np.arange(NEURAL_WINDOW + 1, ctx_len + 1)])
        self.ctx_len = ctx_len

    def _window(self, sentence: str, idx: int, exclude: str):
        toks = [_clean_token(t) for t in sentence.split()]
        left = [self.vocab[t] for t in toks[max(0, idx - NEURAL_WINDOW):idx]
                if t in self.vocab and t != exclude][-NEURAL_WINDOW:]
        right = [self.vocab[t] for t in toks[idx + 1:idx + 1 + NEURAL_WINDOW]
                 if t in self.vocab and t != exclude][:NEURAL_WINDOW]
        ids = np.full(self.ctx_len, self.pad, dtype=np.int64)
        ids[:len(left)] = left
        ids[NEURAL_WINDOW:NEURAL_WINDOW + len(right)] = right
        return ids

    def margins_ctx(self, word_idx, cand_idxs, sentence, idx, exclude):
        ids = self._window(sentence, idx, exclude)
        logits = self.sess.run(None, {"context_ids": ids[None, :],
                                      "pos_ids": self.pos[None, :]})[0][0]
        logp = logits - np.logaddexp.reduce(logits)
        return np.array([logp[c] for c in cand_idxs]) - logp[word_idx]

class _CtxLMScorer:
    """Loads the trainer's .npz and exposes margins(observed, cands, ctx)."""
    def __init__(self, npz_path, vocab_size, alpha=0.4):
        from ctxlm_scorer import load_ctx
        self.uni, self.bg_dict, self.log_p_uni, self.alpha = load_ctx(npz_path, vocab_size, alpha)
    def margins(self, word_idx, cand_idxs, ctx):
        from ctxlm_scorer import _bigram_count
        total = max(int(self.uni.sum()), 1)
        denom = total + len(self.uni)

        def score(t):
            import math
            p_smooth = (int(self.uni[t]) + 1) / denom
            s = 0.0
            for c in ctx:
                if c == t:
                    continue
                c_count = int(self.uni[c])
                cnt = _bigram_count(self.bg_dict, None, c, t)
                # min-support gate: an MLE conditional off a rare context
                # word (c(c)=2, cnt=2 -> P=1.0) is noise, not evidence;
                # fall back to the smoothed unigram below MIN_SUPPORT
                if cnt > 0 and c_count >= MIN_CONTEXT_SUPPORT:
                    p = cnt / c_count
                else:
                    p = self.alpha * p_smooth
                s += math.log(p)
            return s
        # candidate-minus-observed, the polarity every other scorer uses:
        # positive margin = the candidate fits the context BETTER
        base = score(word_idx)
        return np.array([score(w) - base for w in cand_idxs], dtype=np.float64)


SCORERS["ctxlm"] = None  # resolved lazily in main()




def adjacent_ids(sentence: str, idx: int, vocab: dict[str, int]) -> list[int]:
    """The immediately adjacent in-vocab tokens — the context a BIGRAM
    can actually score. Window-summing non-adjacent pairs mostly
    measures backoff fallback, not fit."""
    toks = sentence.split()
    ids = []
    for i in (idx - 1, idx + 1):
        if 0 <= i < len(toks):
            tok = _clean_token(toks[i])
            if tok and tok in vocab:
                ids.append(vocab[tok])
    return ids


def context_ids(sentence: str, idx: int, vocab: dict[str, int], exclude: str) -> list[int]:
    toks = sentence.split()
    ids = []
    for i in range(max(0, idx - WINDOW), min(len(toks), idx + WINDOW + 1)):
        if i == idx:
            continue
        tok = _clean_token(toks[i])
        if tok and tok != exclude and tok in vocab:
            ids.append(vocab[tok])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--scorer", choices=sorted(SCORERS) + ["ctxlm", "neural"], default="cosine")
    parser.add_argument("--clean-sample", type=int, default=4000)
    args = parser.parse_args()

    vocab, raw, normalized = load_tier(args.lang)
    if args.scorer == "skipgram":
        output = np.load(REPO_ROOT / f"eval/cache/{args.lang}.output.npy")
        scorer = SkipgramScorer(output, raw)
    elif args.scorer == "ctxlm":
        from ctxlm_scorer import load_ctx
        scorer = _CtxLMScorer(REPO_ROOT / f"models/{args.lang}/fasttext.{args.lang}.ctx.npz",
                              vocab_size=max(vocab.values()) + 1, alpha=0.4)
    elif args.scorer == "neural":
        scorer = _NeuralScorer(REPO_ROOT / f"models/{args.lang}/fasttext.{args.lang}.ctx-neural.onnx", vocab)
    else:
        scorer = SCORERS[args.scorer](None, normalized)

    confusion = json.loads((REPO_ROOT / f"eval/confusion/{args.lang}.json").read_text())["table"]
    realword = json.loads((REPO_ROOT / f"eval/realword/{args.lang}.json").read_text())
    clean_lines = (REPO_ROOT / f"eval/realword/{args.lang}.clean.jsonl").read_text().splitlines()

    # ---- recall side -------------------------------------------------
    n_instances = covered = 0
    margins_all: list[float] = []    # max over candidates (flag decision)
    margins_true: list[float] = []   # margin of the true correction
    for pair in realword["pairs"]:
        typo, correction = pair["typo"], pair["correction"]
        if typo not in vocab or correction not in vocab:
            continue
        candidates = [w for w in confusion.get(typo, {}) if w in vocab and w != typo]
        true_pos = candidates.index(correction) if correction in candidates else None
        for ctx_info in pair["contexts"]:
            ctx = (adjacent_ids(ctx_info["sentence"], ctx_info["idx"], vocab)
                   if args.scorer == "ctxlm"
                   else context_ids(ctx_info["sentence"], ctx_info["idx"], vocab, typo))
            if not ctx:
                continue
            n_instances += 1
            if true_pos is not None:
                covered += 1
            if not candidates:
                continue
            if hasattr(scorer, "margins_ctx"):
                marg = scorer.margins_ctx(vocab[typo], [vocab[c] for c in candidates],
                                          ctx_info["sentence"], ctx_info["idx"], typo)
            else:
                marg = scorer.margins(vocab[typo], [vocab[c] for c in candidates], ctx)
            margins_all.append(float(marg.max()))
            if true_pos is not None:
                margins_true.append(float(marg[true_pos]))

    # ---- FP side -----------------------------------------------------
    rng = np.random.default_rng(42)
    sample = rng.choice(len(clean_lines), size=min(args.clean_sample, len(clean_lines)), replace=False)
    fp_margins: list[float] = []
    for si in sample:
        sentence = clean_lines[si]
        for idx, raw_tok in enumerate(sentence.split()):
            tok = _clean_token(raw_tok)
            if not tok or tok not in vocab or tok not in confusion:
                continue
            candidates = [w for w in confusion[tok] if w in vocab and w != tok]
            ctx = (adjacent_ids(sentence, idx, vocab)
                   if args.scorer == "ctxlm"
                   else context_ids(sentence, idx, vocab, tok))
            if not candidates or not ctx:
                continue
            if hasattr(scorer, "margins_ctx"):
                marg = scorer.margins_ctx(vocab[tok], [vocab[c] for c in candidates],
                                          sentence, idx, tok)
            else:
                marg = scorer.margins(vocab[tok], [vocab[c] for c in candidates], ctx)
            fp_margins.append(float(marg.max()))

    ma, mt, mf = (np.array(x) if x else np.zeros(0) for x in (margins_all, margins_true, fp_margins))

    def at(tau: float) -> dict:
        return {
            "tau": round(float(tau), 4),
            "flag_rate_scoreable": round(float((ma > tau).mean()), 4) if ma.size else None,
            "recall_true_correction_top": round(float((mt > tau).mean()), 4) if mt.size else None,
            "fp_rate": round(float((mf > tau).mean()), 4) if mf.size else None,
        }

    grid = [at(t) for t in TAUS]
    operating = [at(np.quantile(mf, q)) for q in FP_ANCHORS] if mf.size else []

    doc = {
        "language": args.lang,
        "scorer": args.scorer,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coverage": {
            "n_instances": n_instances,
            "table_covered_instances": covered,
            "table_coverage": round(covered / n_instances, 4) if n_instances else None,
            "scoreable_instances": int(ma.size),
            "clean_eligible_tokens": int(mf.size),
        },
        "operating_points_fp_anchored": operating,
        "grid": grid,
    }
    out = REPO_ROOT / f"eval/realword/{args.lang}.probe.{args.scorer}.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["coverage"]))
    print(f"{'tau':>8} {'flag%':>7} {'true-top%':>9} {'fp%':>6}")
    for r in operating:
        print(f"{r['tau']:>8} {r['flag_rate_scoreable']*100:>6.1f}% {r['recall_true_correction_top']*100:>8.1f}% {r['fp_rate']*100:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
