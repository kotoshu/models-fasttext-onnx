#!/usr/bin/env python3
"""Plan 123: score the fastText tiers on the synthetic typo corpora.

The corpus_bench ranking rule, applied to the dictionary-grounded
synthetic pairs: rank the correction among the tier's whole vocabulary
by cosine to the typo, excluding the typo itself, ties optimistic.
Tiers skip pairs outside their vocabulary (counted — the discipline
the frozen benches use). Corrections outside the full vocabulary are
counted as correction-OOV (the dictionary is bigger than the model
vocab by design — those pairs measure the vocabulary cut, not ranking).

This is the tier-baseline half of the plan-115 evidence: the hybrid
C-retrieve + fastText-rescore comparison rides the same corpora when
the plan-114 training environment is reassembled; its frozen REAL
components remain the ship gate either way.

Usage:
  python3 eval/synth_corpus_bench.py --repo-root . --lang de es
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HIT_KS = (1, 5, 20)


def rank_one(normalized: np.ndarray, typo_idx: int, corr_idx: int) -> int | None:
    sims = normalized @ normalized[typo_idx]
    sims[typo_idx] = -np.inf
    order = np.argsort(-sims, kind="stable")
    pos = int(np.where(order == corr_idx)[0][0])
    return pos + 1


def score_tier(model, pairs: list) -> dict:
    w2i = model.word_to_idx
    hits = {k: 0 for k in HIT_KS}
    n = typo_oov = corr_oov = 0
    for typo, corr, _count in pairs:
        t = w2i.get(typo)
        c = w2i.get(corr)
        if c is None:
            corr_oov += 1
            continue
        if t is None:
            typo_oov += 1
            continue
        rank = rank_one(model.normalized, t, c)
        n += 1
        for k in HIT_KS:
            if rank <= k:
                hits[k] += 1
    out = {
        "pairs_evaluated": n,
        "pairs_typo_oov": typo_oov,
        "pairs_correction_oov": corr_oov,
    }
    for k in HIT_KS:
        out[f"top{k}"] = round(hits[k] / n, 4) if n else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan 123 tier baseline on synthetic corpora")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--lang", nargs="+", default=["de", "es"])
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "eval"))
    import run_eval  # noqa: E402

    out: dict = {"plan": "123 synthetic-corpus tier baseline", "rule": "corpus_bench ranking rule"}
    for lang in args.lang:
        corpus = json.loads((repo / "eval" / "corpora" / "synth" / f"{lang}.json").read_text(encoding="utf-8"))
        model_dir = repo / "models" / lang
        result: dict = {"n_pairs": corpus["n_pairs_unique"], "provenance": corpus["corpus"]}

        full = run_eval.load_full_model(model_dir / f"fasttext.{lang}.onnx", model_dir / f"fasttext.{lang}.vocab.json")
        t0 = time.perf_counter()
        result["full"] = score_tier(full, corpus["pairs"])
        print(f"{lang} full: {result['full']} ({time.perf_counter() - t0:.1f}s)")

        for tier in ("fluency", "mini"):
            tier_model = run_eval.load_tier_model(
                model_dir / f"fasttext.{lang}.{tier}.onnx", model_dir / f"fasttext.{lang}.{tier}.vocab.json"
            )
            result[tier] = score_tier(tier_model, corpus["pairs"])
            print(f"{lang} {tier}: {result[tier]}")

        out[lang] = result

    dest = repo / "eval" / "reports" / "synth-corpus.summary.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
