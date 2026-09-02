#!/usr/bin/env python3
"""Real-typo corpus benchmark for the tier models (report-only, no gates).

Uses (typo -> correction) pairs extracted from the GitHub Typo Corpus
(Hagiwara & Mita, LREC 2020; see scripts/fetch_corpus.py) to measure how
well each model ranks the human correction among all its vocabulary words
by cosine similarity to the typo.

Per language and per model tier (full / fluency / mini):
- keep pairs where BOTH words are in the FULL vocabulary (the embedding
  side can only be evaluated on words it can embed);
- among those, a tier additionally skips pairs whose typo or correction is
  outside ITS vocabulary (counted as typo_oov / correction_oov);
- for every remaining pair, rank the correction among all tier-vocab words
  by cosine-to-typo similarity and record top-1 / top-5 / top-20 hits.

Determinism: pairs are deduplicated upstream; if more than --max-pairs
remain they are subsampled with the same RNG discipline as run_eval.py
(np.random.default_rng([seed, crc32(language)])), so a given corpus +
model pair always yields the same report apart from the timestamp.

Output: eval/reports/corpus.{lang}.json (committed; the corpus itself is
never committed).

Usage:
  python3 eval/corpus_bench.py --all --repo-root .
  python3 eval/corpus_bench.py --lang en de --repo-root .
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 42
MAX_PAIRS_DEFAULT = 2000
HIT_KS = (1, 5, 20)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def manifest_languages(repo: Path) -> list[str]:
    manifest = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))
    return sorted({e["language"] for e in manifest["resources"].values() if e.get("type") == "onnx"})


def _rank_stats(ranks: np.ndarray) -> dict:
    return {
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def bench_language(repo: Path, run_eval, lang: str, max_pairs: int) -> dict:
    import numpy as _np

    model_dir = repo / "models" / lang
    full = run_eval.load_full_model(
        model_dir / f"fasttext.{lang}.onnx", model_dir / f"fasttext.{lang}.vocab.json"
    )
    tiers = {"full": full}
    for tier in ("fluency", "mini"):
        tiers[tier] = run_eval.load_tier_model(
            model_dir / f"fasttext.{lang}.{tier}.onnx",
            model_dir / f"fasttext.{lang}.{tier}.vocab.json",
        )

    corpus_path = repo / "eval" / "corpora" / f"{lang}.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    extracted = corpus["n_pairs_unique"]

    # both words in the FULL vocabulary (deduplicated upstream)
    pool = [
        (t, c)
        for t, c, _count in corpus["pairs"]
        if t != c and t in full.word_to_idx and c in full.word_to_idx
    ]
    in_full = len(pool)
    if len(pool) > max_pairs:
        rng = np.random.default_rng([SEED, zlib.crc32(lang.encode("utf-8"))])
        pool = [pool[i] for i in rng.permutation(len(pool))[:max_pairs]]
    sampled = len(pool)

    results: dict[str, dict] = {}
    for tier_name, model in tiers.items():
        hits = {k: 0 for k in HIT_KS}
        ranks: list[int] = []
        typo_oov = 0
        corr_oov = 0
        for typo, correction in pool:
            t = model.word_to_idx.get(typo)
            c = model.word_to_idx.get(correction)
            if t is None:
                typo_oov += 1
                continue
            if c is None:
                corr_oov += 1
                continue
            sims = model.normalized @ model.normalized[t]
            sims[t] = -np.inf  # the typo is the query; never suggest it back
            s = sims[c]
            rank0 = int(_np.count_nonzero(sims > s))  # 0-based; ties optimistic
            ranks.append(rank0 + 1)
            for k in HIT_KS:
                if rank0 < k:
                    hits[k] += 1
        n = len(ranks)
        results[tier_name] = {
            "vocab_size": model.vocab_size,
            "pairs_evaluated": n,
            "pairs_typo_oov": typo_oov,
            "pairs_correction_oov": corr_oov,
            "top1": hits[1] / n if n else None,
            "top5": hits[5] / n if n else None,
            "top20": hits[20] / n if n else None,
            **(_rank_stats(np.asarray(ranks)) if n else {}),
            "ranking": "rank of the correction among the tier's whole vocabulary by cosine to the typo, excluding the typo itself (ties counted optimistically)",
        }

    return {
        "language": lang,
        "corpus": dict(corpus["corpus"], extraction=corpus["extraction"]),
        "pairs": {
            "extracted_unique": extracted,
            "both_in_full_vocab": in_full,
            "sampled_for_eval": sampled,
            "sampling": (
                f"first {max_pairs} of default_rng([{SEED}, crc32(language)]) permutation"
                if in_full > max_pairs
                else "all pairs (below cap)"
            ),
        },
        "tiers": results,
        "determinism": {"seed": SEED, "rng": "np.random.default_rng([seed, crc32(language)])"},
        "report_only": True,
        "generated_at": iso_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Corpus (real-typo) benchmark, report-only")
    parser.add_argument("--lang", nargs="+", help="language codes, e.g. en de")
    parser.add_argument("--all", action="store_true", help="every manifest language with extracted pairs")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_DEFAULT, help=f"cap on pairs per language (default {MAX_PAIRS_DEFAULT})")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "eval"))
    import run_eval  # noqa: E402

    if args.all:
        langs = [
            l
            for l in manifest_languages(repo)
            if (repo / "eval" / "corpora" / f"{l}.json").exists()
        ]
        if not langs:
            parser.error("no extracted corpora found — run scripts/fetch_corpus.py first")
    elif args.lang:
        langs = args.lang
    else:
        parser.error("specify --lang or --all")

    reports_dir = repo / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        report = bench_language(repo, run_eval, lang, args.max_pairs)
        out = reports_dir / f"corpus.{lang}.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        p = report["pairs"]
        line = f"{lang}: pairs {p['extracted_unique']} extracted / {p['both_in_full_vocab']} in-vocab / {p['sampled_for_eval']} eval"
        for tier_name, t in report["tiers"].items():
            line += (
                f" | {tier_name}: n={t['pairs_evaluated']}"
                f" top1={t['top1']:.3f} top5={t['top5']:.3f} top20={t['top20']:.3f}"
                if t["pairs_evaluated"]
                else f" | {tier_name}: n=0"
            )
        print(line + f" -> {out.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
