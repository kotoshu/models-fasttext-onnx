#!/usr/bin/env python3
"""Plan 114: score candidate C v2 against the fastText tiers on the frozen
C-benchmark (report-only, no gates).

Bench: eval/cbench/cbench.{lang}.json (frozen by scripts/build_cbench.py,
receipt at eval/reports/cbench.frozen.json). Two labeled components per
language: REAL (held-out-repo-clean corpus pairs; both words in the full
vocabulary) and SYNTH (noise.py generator pairs for de/ru/es; generator-
domain evidence only — v2 trains on the same generator, so the synth
component carries a train-distribution advantage for C by construction; the
real component is the decision evidence).

Scoring, per language and component:
- fastText tiers (full / fluency / mini): the exact corpus_bench rule — rank
  the correction among the tier's whole vocabulary by cosine to the typo,
  excluding the typo itself, ties optimistic; tiers skip pairs outside their
  own vocabulary (counted, corpus_bench discipline);
- candidate C v1 (real-corpus-only training, 80/20 split): same rule over the
  full 100k vocabulary under the C char encoder — the no-synth/no-de-training
  ablation and v1-transfer comparator. v1 trains on the first 80% of the same
  repo permutation, so bench pairs confined to the perm[70%:80%] repos are
  v1-TRAIN-tainted: v1 rows skip them (counted as excluded);
- candidate C v2 (plan 114: 70/30 split + real + synth de/ru/es training):
  same rule;
- hybrid C-retrieve + fastText-rescore (the bake-off top-1 thread): candidates
  = C v2 top-20, rescored by fastText FULL cosine(typo, candidate); the
  correction outside the top-20 is a miss for every k (rank 21). Measures
  whether C recall + fastText precision fixes C's weak top-1.

Latency: batch-1 per-suggest median/p90 on REAL bench queries (C v2 query
encode + matvec over the amortized vocab matrix; hybrid adds the fastText
rescore of 20 candidates), same measurement style as eval/bakeoff_bench.py.

Output: eval/reports/biencoder-v2.{lang}.json + eval/reports/biencoder-v2.summary.json
with the pre-declared plan-114 decision rule and the measured verdict inputs.

Usage:
  python3 eval/cbench_bench.py --repo-root . [--lang en de ru es]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

SEED = 42
HIT_KS = (1, 5, 20)
LANGS_DEFAULT = ("en", "de", "ru", "es")
HYBRID_TOPK = 20
LATENCY_QUERIES = 200
V1_SPLIT_TRAIN_FRAC = 0.8  # scripts/train_typo_biencoder.py (bakeoff) split


def v1_train_taint_mask(repo: Path) -> set[tuple[str, str]]:
    """String pairs seen in any v1 (80/20) TRAIN repo — tainted for the v1 comparator.

    v1 trains on the FIRST 80% of the same default_rng([42]) repo permutation
    v2/bench use for 70/30, so repos perm[70%:80%] are v1-train but v2-clean;
    bench pairs confined to that band would be memorized by v1. The v1 rows
    below evaluate only on pairs this function does NOT return.
    """
    import zlib

    sys.path.insert(0, str(repo / "scripts"))
    from fetch_corpus import LANG_MAP, _word_pair

    pair_repos: dict[tuple[str, str, str], set[str]] = {}
    with gzip.open(repo / "eval" / "corpora" / "github-typo-corpus.v1.0.0.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            url = obj.get("repo", "")
            for edit in obj.get("edits", []):
                src = edit.get("src") or {}
                lang = LANG_MAP.get(src.get("lang", ""))
                if lang is None:
                    continue
                tgt = edit.get("tgt") or {}
                st, tt = src.get("text", "").split(), tgt.get("text", "").split()
                if not st or len(st) != len(tt):
                    continue
                diffs = [(a, b) for a, b in zip(st, tt) if a != b]
                if len(diffs) != 1:
                    continue
                pair = _word_pair(*diffs[0])
                if pair is None:
                    continue
                pair_repos.setdefault((lang, pair[0], pair[1]), set()).add(url)
    repos = sorted({r for rs in pair_repos.values() for r in rs})
    rng = np.random.default_rng([SEED])
    perm = rng.permutation(len(repos))
    v1_train = {repos[i] for i in perm[: int(len(repos) * V1_SPLIT_TRAIN_FRAC)]}
    return {(t, c) for (l, t, c), rs in pair_repos.items() if rs & v1_train}

# pre-declared plan-114 ship rule (opt-in registry resource)
CLEAR_WIN_PP = 5.0     # top-5 margin over the full tier, percentage points
DEGRADE_PP = 2.0       # allowed regression before a component counts as degraded
SIZE_CAP_MB = 30.0
LATENCY_CAP_MS = 100.0


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalized(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class CharEncoder:
    """Candidate C int8 ONNX char bi-encoder (v1 or v2 artifact dir)."""

    MAX_WORD_LEN = 24

    def __init__(self, cand_dir: Path) -> None:
        self.char_to_idx = json.loads((cand_dir / "char_to_idx.json").read_text(encoding="utf-8"))
        self.unk = len(self.char_to_idx) + 1
        self.sess = ort.InferenceSession(str(cand_dir / "model.int8.onnx"), providers=["CPUExecutionProvider"])

    def ids_for(self, words: list[str]) -> np.ndarray:
        ids = np.zeros((len(words), self.MAX_WORD_LEN), dtype=np.int64)
        for i, w in enumerate(words):
            for j, ch in enumerate(w[: self.MAX_WORD_LEN]):
                ids[i, j] = self.char_to_idx.get(ch, self.unk)
        return ids

    def encode(self, words: list[str], batch: int = 4096) -> np.ndarray:
        chunks = []
        for lo in range(0, len(words), batch):
            chunks.append(self.sess.run(["embedding"], {"char_ids": self.ids_for(words[lo : lo + batch])})[0])
        return np.vstack(chunks).astype(np.float32)


def rank_bench(normalized: np.ndarray, word_to_idx: dict, pool: list) -> dict:
    """corpus_bench ranking rule over an embedding matrix (typo must be in vocab)."""
    hits = {k: 0 for k in HIT_KS}
    ranks: list[int] = []
    typo_oov = 0
    corr_oov = 0
    for typo, correction in pool:
        t = word_to_idx.get(typo)
        c = word_to_idx.get(correction)
        if t is None:
            typo_oov += 1
            continue
        if c is None:
            corr_oov += 1
            continue
        sims = normalized @ normalized[t]
        sims[t] = -np.inf
        s = sims[c]
        rank0 = int(np.count_nonzero(sims > s))
        ranks.append(rank0 + 1)
        for k in HIT_KS:
            if rank0 < k:
                hits[k] += 1
    n = len(ranks)
    return {
        "pairs_evaluated": n,
        "pairs_typo_oov": typo_oov,
        "pairs_correction_oov": corr_oov,
        "top1": hits[1] / n if n else None,
        "top5": hits[5] / n if n else None,
        "top20": hits[20] / n if n else None,
        "mean_rank": float(np.mean(ranks)) if n else None,
        "median_rank": float(np.median(ranks)) if n else None,
        "ranking": (
            "rank of the correction among the whole vocabulary by cosine to the typo, "
            "excluding the typo itself (ties counted optimistically) — corpus_bench rule"
        ),
    }


def hybrid_bench(c_nrm: np.ndarray, ft_nrm: np.ndarray, word_to_idx: dict, pool: list, topk: int) -> dict:
    """C-retrieve top-k, fastText-full rescore; correction outside slate = miss for every k."""
    hits = {k: 0 for k in HIT_KS}
    ranks: list[int] = []
    for typo, correction in pool:
        t = word_to_idx.get(typo)
        c = word_to_idx.get(correction)
        if t is None or c is None:
            continue  # oov already counted by the callers' rule; hybrid reuses in-vocab pairs
        sims = c_nrm @ c_nrm[t]
        sims[t] = -np.inf
        slate = np.argpartition(-sims, topk - 1)[:topk]
        if c not in slate:
            ranks.append(topk + 1)
            continue
        ft_sims = ft_nrm[slate] @ ft_nrm[t]
        corr_pos = int(np.where(slate == c)[0][0])
        rank0 = int(np.count_nonzero(ft_sims > ft_sims[corr_pos]))
        ranks.append(rank0 + 1)
        for k in HIT_KS:
            if rank0 < k:
                hits[k] += 1
    n = len(ranks)
    return {
        "pairs_evaluated": n,
        "slate": f"C top-{topk} rescored by fastText full cosine to the typo",
        "top1": hits[1] / n if n else None,
        "top5": hits[5] / n if n else None,
        "top20": hits[20] / n if n else None,
        "mean_rank": float(np.mean(ranks)) if n else None,
        "median_rank": float(np.median(ranks)) if n else None,
    }


def _bootstrap_ci(deltas: list[int], n_boot: int = 2000) -> list[float]:
    """95% CI of the mean of paired win/loss indicators (1/0 hits)."""
    if not deltas:
        return [0.0, 0.0]
    arr = np.asarray(deltas, dtype=np.float32)
    rng = np.random.default_rng([SEED, 13])
    means = [float(np.mean(arr[rng.integers(len(arr), size=len(arr))])) for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_top5(c_nrm, ft_nrm, word_to_idx, pool) -> dict:
    """Paired per-pair top-5 comparison of C vs the full tier (win = C hits, ft misses)."""
    wins = losses = both = neither = 0
    deltas: list[int] = []
    for typo, correction in pool:
        t = word_to_idx.get(typo)
        c = word_to_idx.get(correction)
        if t is None or c is None:
            continue
        sc = c_nrm @ c_nrm[t]
        sc[t] = -np.inf
        sf = ft_nrm @ ft_nrm[t]
        sf[t] = -np.inf
        c_hit = int(np.count_nonzero(sc > sc[c]) < 5)
        f_hit = int(np.count_nonzero(sf > sf[c]) < 5)
        deltas.append(c_hit - f_hit)
        if c_hit and f_hit:
            both += 1
        elif c_hit:
            wins += 1
        elif f_hit:
            losses += 1
        else:
            neither += 1
    ci = _bootstrap_ci(deltas)
    return {
        "c_only_wins": wins,
        "ft_only_wins": losses,
        "both_hit": both,
        "neither": neither,
        "net_pp": 100.0 * float(np.mean(deltas)) if deltas else None,
        "ci95_pp": [100.0 * ci[0], 100.0 * ci[1]],
    }


def _cached_matrix(cache_path: Path, encode_fn) -> np.ndarray:
    """Cache the per-language vocabulary embedding matrix (gitignored dir).

    Same pattern as eval/bakeoff_bench.py: the cache key is the artifact
    directory name + language; rebuilt artifacts recreate their directory and
    move the cache path with them.
    """
    if cache_path.exists():
        return np.load(cache_path)
    mat = encode_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mat)
    return mat


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan 114 C-benchmark bench, report-only")
    parser.add_argument("--lang", nargs="+", default=list(LANGS_DEFAULT))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "eval"))
    import run_eval  # noqa: E402

    frozen = json.loads((repo / "eval" / "reports" / "cbench.frozen.json").read_text(encoding="utf-8"))
    manifest = json.loads((repo / "eval" / "candidates" / "manifest.json").read_text(encoding="utf-8"))
    c_v1 = CharEncoder(repo / "eval" / "candidates" / "c_typo")
    c_v2 = CharEncoder(repo / "eval" / "candidates" / "c_typo_v2")

    print("streaming corpus for the v1 taint mask (80/20 split)...")
    v1_taint = v1_train_taint_mask(repo)
    print(f"  v1 train string pairs: {len(v1_taint)}")

    reports_dir = repo / "eval" / "reports"
    summary_langs: dict[str, dict] = {}

    for lang in args.lang:
        print(f"== {lang}")
        bench = json.loads((repo / "eval" / "cbench" / f"cbench.{lang}.json").read_text(encoding="utf-8"))
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
        words = [None] * full.vocab_size
        for w, i in full.word_to_idx.items():
            words[i] = w

        print(f"  C embedding {full.vocab_size} vocab words (v1, v2)")
        cache_dir = repo / "eval" / "candidates" / "cache"
        c1_nrm = _normalized(_cached_matrix(cache_dir / f"c_typo.vocab.{lang}.npy", lambda: c_v1.encode(words)))
        c2_nrm = _normalized(_cached_matrix(cache_dir / f"c_typo_v2.vocab.{lang}.npy", lambda: c_v2.encode(words)))
        print("    vocab embedding done (cached)")

        components: dict[str, dict] = {}
        for comp in ("real", "synth"):
            pool = [tuple(p) for p in bench[comp]]
            if not pool:
                continue
            res: dict[str, dict] = {}
            for tier_name, model in tiers.items():
                res[tier_name] = rank_bench(model.normalized, model.word_to_idx, pool)
            res["c_v1"] = rank_bench(c1_nrm, full.word_to_idx, [p for p in pool if p not in v1_taint])
            res["c_v1"]["pairs_excluded_v1_train_tainted"] = sum(1 for p in pool if p in v1_taint)
            res["c_v2"] = rank_bench(c2_nrm, full.word_to_idx, pool)
            res["hybrid_c2_top20_ft_rescore"] = hybrid_bench(c2_nrm, full.normalized, full.word_to_idx, pool, HYBRID_TOPK)
            res["c_v2_vs_full_paired_top5"] = paired_top5(c2_nrm, full.normalized, full.word_to_idx, pool)
            components[comp] = {"pairs": len(pool), **res}
            f5 = res["full"]["top5"]
            v5 = res["c_v2"]["top5"]
            print(
                f"  {comp}: n={len(pool)} | full top5={f5:.3f} | c_v1 {res['c_v1']['top5']:.3f}"
                f" | c_v2 {v5:.3f} ({(v5 - f5) * 100:+.1f} pp, paired net"
                f" {res['c_v2_vs_full_paired_top5']['net_pp']:+.1f} pp)"
                f" | hybrid top1={res['hybrid_c2_top20_ft_rescore']['top1']:.3f} vs full {res['full']['top1']:.3f}"
            )

        # batch-1 latency on real queries (amortized vocab matrices)
        lat_pool = [tuple(p) for p in bench["real"]][:LATENCY_QUERIES]
        samples_v2 = []
        samples_hyb = []
        for typo, _c in lat_pool:
            t0 = time.perf_counter()
            q = c_v2.sess.run(["embedding"], {"char_ids": c_v2.ids_for([typo])})[0]
            sims = c2_nrm @ (q[0] / max(float(np.linalg.norm(q[0])), 1e-9))
            np.argpartition(-sims, 20)[:20]
            samples_v2.append((time.perf_counter() - t0) * 1e3)
        for typo, _c in lat_pool:
            t = full.word_to_idx.get(typo)
            if t is None:
                continue
            t0 = time.perf_counter()
            q = c_v2.sess.run(["embedding"], {"char_ids": c_v2.ids_for([typo])})[0]
            sims = c2_nrm @ (q[0] / max(float(np.linalg.norm(q[0])), 1e-9))
            sims[t] = -np.inf
            slate = np.argpartition(-sims, HYBRID_TOPK - 1)[:HYBRID_TOPK]
            ft_sims = full.normalized[slate] @ full.normalized[t]
            np.argpartition(-ft_sims, 5)[:5]
            samples_hyb.append((time.perf_counter() - t0) * 1e3)

        def stats(samples: list[float]) -> dict:
            arr = np.asarray(samples)
            return {"n": len(arr), "median_ms": float(np.median(arr)), "p90_ms": float(np.percentile(arr, 90))}

        latency = {
            "c_v2_query_plus_matvec": stats(samples_v2),
            "hybrid_c2_top20_ft_rescore": stats(samples_hyb),
            "note": "amortized per-language vocab embedding matrix; incremental per-suggest cost",
        }

        report = {
            "language": lang,
            "protocol": (
                "frozen plan-114 C-benchmark; tiers scored with the exact corpus_bench rule; "
                "C v1/v2 rank the correction among the full 100k vocabulary under the char encoder; "
                "synth = noise.py generator domain (v2 trains on the same generator — carries a "
                "train-distribution advantage for C; real is the decision evidence)"
            ),
            "frozen": {"languages": frozen["languages"][lang], "corpus_sha256": frozen["corpus_sha256"]},
            "components": components,
            "latency_ms_per_suggest": latency,
            "models": {
                "c_v1": manifest["c_typo"],
                "c_v2": manifest["c_typo_v2"],
            },
            "report_only": True,
            "generated_at": iso_now(),
        }
        out = reports_dir / f"biencoder-v2.{lang}.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"-> {out.relative_to(repo)}")

        summary_langs[lang] = {
            "real_pairs": len(bench["real"]),
            "synth_pairs": len(bench["synth"]),
            "real": {
                t: {k: components["real"][t][k] for k in ("top1", "top5", "top20", "pairs_evaluated")}
                for t in ("full", "fluency", "mini", "c_v1", "c_v2", "hybrid_c2_top20_ft_rescore")
            },
            **({"synth": {
                t: {k: components["synth"][t][k] for k in ("top1", "top5", "top20", "pairs_evaluated")}
                for t in ("full", "fluency", "mini", "c_v1", "c_v2", "hybrid_c2_top20_ft_rescore")
            }} if "synth" in components else {}),
            "c_v2_vs_full_paired_top5": {
                comp: components[comp]["c_v2_vs_full_paired_top5"] for comp in components
            },
            "latency_median_ms": {
                "c_v2": latency["c_v2_query_plus_matvec"]["median_ms"],
                "hybrid": latency["hybrid_c2_top20_ft_rescore"]["median_ms"],
            },
        }

    summary = {
        "plan": "114 typo bi-encoder v2",
        "decision_rule": {
            "rule": (
                "ship C v2 as an OPT-IN registry resource only if it beats the fastText FULL tier "
                f"top-5 by >= {CLEAR_WIN_PP:g} pp (paired net, CI excluding 0) on en AND de, and "
                f"degrades nowhere by more than {DEGRADE_PP:g} pp top-5 on any language/component "
                f"with n >= 20 real pairs (synth components: generator-domain evidence, labeled); "
                f"AND size <= {SIZE_CAP_MB:g} MB int8 AND median per-suggest latency "
                f"<= {LATENCY_CAP_MS:g} ms. An honest reject is a complete outcome (int4 precedent); "
                "no registry change, no tag on a reject"
            ),
        },
        "bench": {lang: frozen["languages"][lang] for lang in summary_langs},
        "languages": summary_langs,
        "models": {
            "c_v1_int8_mb": manifest["c_typo"]["int8_mb"],
            "c_v2_int8_mb": manifest["c_typo_v2"]["int8_mb"],
        },
        "hardware": "Apple M1 Max (arm64), CPUExecutionProvider",
        "generated_at": iso_now(),
    }
    out = reports_dir / "biencoder-v2.summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
