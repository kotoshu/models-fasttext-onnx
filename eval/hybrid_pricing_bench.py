#!/usr/bin/env python3
"""Plan 115: price the hybrid retrieval thread before believing it.

The plan-114 reject left one survivor on record: the hybrid (C v2 top-20
retrieval + fastText FULL rescore) beat the full tier top-1 AND top-5 on
ALL FOUR real C-benchmark components. Before any ship decision this bench
prices its real cost (the per-language 100k x 256 retrieval matrix) and
re-scores it on REAL pairs only.

Pre-declared decision rule (plan-114 discipline, adapted to the hybrid and
binding on the shippable int8 configuration):

(a) CLEAR WIN: the hybrid beats the fastText FULL tier top-5 by >= 5 pp
    (paired net, 95% CI excluding 0) on the en AND de real components;
(b) DEGRADES NOWHERE: no real component with n >= 20 pairs loses more
    than 2 pp top-5 vs the full tier;
(c) PRICE: incremental artifact (C v2 int8 model 0.481 MB + per-language
    retrieval matrix, int8 variant incl. fp16 row scales) <= 30 MB, and
    median per-suggest latency <= 100 ms on the int8 path.

SHIP as an opt-in registry resource only on (a) AND (b) AND (c); honest
reject otherwise. No registry change and no tag on a reject. Synth
components are NOT part of the verdict (plan-114: generator-domain
evidence only — v2 trains on the same generator). fp32 brute force and
fp16 matrix variants are measured as quality/price reference points; only
int8 fits the 30 MB cap (fp16 alone is 51.2 MB per language).

Measurements per language (en de ru es):
- matrix pricing: build time (fresh timed encode of the whole 100k vocab
  through the int8 ONNX encoder, warm session), sizes fp32/fp16/int8,
  quantize/dequantize-load times, round-trip distortion;
- retrieval quality vs the fp32 brute-force sweep on the frozen
  C-benchmark REAL component: pure-C hit@k, hybrid hit@k, and
  correction-in-slate@20 agreement;
- hybrid vs full tier on REAL components only: top-1/top-5/top-20 hit
  rates, paired net with 95% bootstrap CIs;
- latency: batch-1 per-suggest on real queries — amortized fp32 path,
  amortized int8 (dequantized at load) path, and a RAM-lean per-query
  chunked-dequant int8 path; retrieval-only vs full suggest.

Output: eval/reports/hybrid-pricing.json (machine receipt; the human
report eval/reports/hybrid-pricing.md quotes it).

Usage:
  PYTHONUNBUFFERED=1 python3 eval/hybrid_pricing_bench.py --repo-root .
"""

from __future__ import annotations

import argparse
import hashlib
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
N_BOOT = 2000

# pre-declared plan-115 ship rule (opt-in registry resource)
CLEAR_WIN_PP = 5.0     # top-5 margin over the full tier, percentage points
DEGRADE_PP = 2.0       # allowed regression before a component counts as degraded
SIZE_CAP_MB = 30.0     # incremental artifact: model int8 + retrieval matrix int8
LATENCY_CAP_MS = 100.0


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalized(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class CharEncoder:
    """Candidate C v2 int8 ONNX char bi-encoder (eval/candidates/c_typo_v2)."""

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


# ---------------------------------------------------------------------------
# quantized retrieval-matrix variants


def quant_int8_per_row(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8: scale = max|row| / 127, scales stored fp16.

    Same shape contract as the repo int4-per-row tiers (one fp16 scale per
    row, kotoshu-rs RowFormat precedent).
    """
    scale = np.max(np.abs(mat), axis=1) / 127.0
    scale[scale == 0.0] = 1.0
    q = np.rint(mat / scale[:, None]).astype(np.int8)
    return q, scale.astype(np.float16)


def dequant_int8(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scales.astype(np.float32)[:, None]


def roundtrip_stats(orig: np.ndarray, recon: np.ndarray) -> dict:
    cos = np.sum(orig * recon, axis=1) / (
        np.linalg.norm(orig, axis=1) * np.maximum(np.linalg.norm(recon, axis=1), 1e-12)
    )
    return {
        "max_abs_err": float(np.max(np.abs(orig - recon))),
        "mean_abs_err": float(np.mean(np.abs(orig - recon))),
        "mean_one_minus_cos": float(np.mean(1.0 - cos)),
    }


def int8_matvec_lean(q_row: np.ndarray, qmat: np.ndarray, scales: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """RAM-lean retrieval: dequantize per query in chunks, never materialize the fp32 matrix."""
    v = qmat.shape[0]
    sims = np.empty(v, dtype=np.float32)
    sc = scales.astype(np.float32)
    for lo in range(0, v, chunk):
        d = qmat[lo : lo + chunk].astype(np.float32)
        d *= sc[lo : lo + chunk, None]
        sims[lo : lo + chunk] = d @ q_row
    return sims


# ---------------------------------------------------------------------------
# scoring (corpus_bench rule, per-pair hit indicators for paired stats)


def rank_bench(normalized: np.ndarray, word_to_idx: dict, pool: list) -> dict:
    """corpus_bench ranking rule; also returns per-pair hit indicators."""
    hits = {k: 0 for k in HIT_KS}
    pair_hits: dict[int, list[int]] = {k: [] for k in HIT_KS}
    typo_oov = corr_oov = 0
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
        for k in HIT_KS:
            hit = int(rank0 < k)
            hits[k] += hit
            pair_hits[k].append(hit)
    n = len(pair_hits[1])
    return {
        "pairs_evaluated": n,
        "pairs_typo_oov": typo_oov,
        "pairs_correction_oov": corr_oov,
        "top1": hits[1] / n if n else None,
        "top5": hits[5] / n if n else None,
        "top20": hits[20] / n if n else None,
        "_pair_hits": pair_hits,
        "ranking": (
            "rank of the correction among the whole vocabulary by cosine to the typo, "
            "excluding the typo itself (ties counted optimistically) — corpus_bench rule"
        ),
    }


def hybrid_bench(c_nrm: np.ndarray, ft_nrm: np.ndarray, word_to_idx: dict, pool: list, topk: int) -> dict:
    """C-retrieve top-k, fastText-full rescore; correction outside slate = miss for every k."""
    hits = {k: 0 for k in HIT_KS}
    pair_hits: dict[int, list[int]] = {k: [] for k in HIT_KS}
    in_slate: list[int] = []
    for typo, correction in pool:
        t = word_to_idx.get(typo)
        c = word_to_idx.get(correction)
        if t is None or c is None:
            continue  # oov counted by the full-tier row; hybrid reuses in-vocab pairs
        sims = c_nrm @ c_nrm[t]
        sims[t] = -np.inf
        slate = np.argpartition(-sims, topk - 1)[:topk]
        corr_in = bool(np.any(slate == c))
        in_slate.append(int(corr_in))
        if not corr_in:
            for k in HIT_KS:
                pair_hits[k].append(0)
            continue
        ft_sims = ft_nrm[slate] @ ft_nrm[t]
        corr_pos = int(np.where(slate == c)[0][0])
        rank0 = int(np.count_nonzero(ft_sims > ft_sims[corr_pos]))
        for k in HIT_KS:
            hit = int(rank0 < k)
            hits[k] += hit
            pair_hits[k].append(hit)
    n = len(pair_hits[1])
    return {
        "pairs_evaluated": n,
        "slate": f"C top-{topk} rescored by fastText full cosine to the typo",
        "top1": hits[1] / n if n else None,
        "top5": hits[5] / n if n else None,
        "top20": hits[20] / n if n else None,
        "recall_at_slate": float(np.mean(in_slate)) if in_slate else None,
        "_pair_hits": pair_hits,
        "_in_slate": in_slate,
    }


def _bootstrap_ci(deltas: list[float]) -> list[float]:
    """95% bootstrap CI of the mean of paired deltas (plan-114 protocol)."""
    if not deltas:
        return [0.0, 0.0]
    arr = np.asarray(deltas, dtype=np.float32)
    rng = np.random.default_rng([SEED, 13])
    means = [float(np.mean(arr[rng.integers(len(arr), size=len(arr))])) for _ in range(N_BOOT)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_hybrid_vs_full(hyb: dict, full: dict, ks: tuple[int, ...] = HIT_KS) -> dict:
    """Paired per-pair comparison of the hybrid vs the full tier (shared in-vocab pairs)."""
    out = {}
    for k in ks:
        h = hyb["_pair_hits"][k]
        f = full["_pair_hits"][k]
        deltas = [a - b for a, b in zip(h, f)]
        ci = _bootstrap_ci(deltas)
        out[f"top{k}"] = {
            "hybrid_only_wins": sum(1 for d in deltas if d > 0),
            "full_only_wins": sum(1 for d in deltas if d < 0),
            "net_pp": 100.0 * float(np.mean(deltas)) if deltas else None,
            "ci95_pp": [100.0 * ci[0], 100.0 * ci[1]],
        }
    return out


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan 115 hybrid retrieval pricing bench")
    parser.add_argument("--lang", nargs="+", default=list(LANGS_DEFAULT))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "eval"))
    import run_eval  # noqa: E402

    frozen = json.loads((repo / "eval" / "reports" / "cbench.frozen.json").read_text(encoding="utf-8"))
    manifest = json.loads((repo / "eval" / "candidates" / "manifest.json").read_text(encoding="utf-8"))
    enc = CharEncoder(repo / "eval" / "candidates" / "c_typo_v2")

    # integrity: the frozen bench files must match the freeze receipt hashes
    for lang, meta in frozen["languages"].items():
        got = file_sha256(repo / meta["file"])
        if got != meta["sha256"]:
            raise SystemExit(f"cbench.{lang}.json sha256 {got} != frozen receipt {meta['sha256']}")
    print("frozen C-benchmark integrity: all sha256 match the receipt")

    cache_dir = repo / "eval" / "candidates" / "cache"
    langs_out: dict[str, dict] = {}

    for lang in args.lang:
        print(f"== {lang}")
        bench = json.loads((repo / "eval" / "cbench" / f"cbench.{lang}.json").read_text(encoding="utf-8"))
        model_dir = repo / "models" / lang
        full = run_eval.load_full_model(
            model_dir / f"fasttext.{lang}.onnx", model_dir / f"fasttext.{lang}.vocab.json"
        )
        words = [None] * full.vocab_size
        for w, i in full.word_to_idx.items():
            words[i] = w

        # ---- build: fresh timed encode of the whole vocabulary (warm session)
        enc.encode(words[:64])  # warmup
        t0 = time.perf_counter()
        built = enc.encode(words)
        build_s = time.perf_counter() - t0
        cache_path = cache_dir / f"c_typo_v2.vocab.{lang}.npy"
        if cache_path.exists():
            cached = np.load(cache_path)
            if not np.allclose(cached, built, atol=1e-5):
                raise SystemExit(f"{cache_path}: cached plan-114 matrix diverges from fresh encode")
            print(f"  build: {build_s:.1f}s for {full.vocab_size} words (parity with plan-114 cache)")
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, built)
            print(f"  build: {build_s:.1f}s (cache was missing, saved)")

        fp32_nrm = _normalized(built)  # the plan-114 brute-force sweep, exactly

        # ---- quantized variants
        t0 = time.perf_counter()
        fp16_stored = fp32_nrm.astype(np.float16)
        fp16_quant_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        fp16_loaded = _normalized(fp16_stored.astype(np.float32))
        fp16_load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        q8, s8 = quant_int8_per_row(fp32_nrm)
        int8_quant_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        int8_loaded = _normalized(dequant_int8(q8, s8))
        int8_load_s = time.perf_counter() - t0

        v = full.vocab_size
        pricing = {
            "vocab_size": v,
            "dims": 256,
            "bytes_fp32": v * 256 * 4,
            "bytes_fp16": v * 256 * 2,
            "bytes_int8_incl_fp16_scales": v * 256 + v * 2,
            "mb_fp32": v * 256 * 4 / 1e6,
            "mb_fp16": v * 256 * 2 / 1e6,
            "mb_int8": (v * 256 + v * 2) / 1e6,
            "build_time_s_encode_100k": round(build_s, 2),
            "build_words_per_s": int(v / build_s),
            "quant_time_s": {"fp16": round(fp16_quant_s, 3), "int8": round(int8_quant_s, 3)},
            "load_dequant_time_s": {"fp16": round(fp16_load_s, 3), "int8": round(int8_load_s, 3)},
            "roundtrip_vs_fp32": {
                "fp16": roundtrip_stats(fp32_nrm, fp16_loaded),
                "int8": roundtrip_stats(fp32_nrm, int8_loaded),
            },
        }
        print(
            f"  pricing: fp32 {pricing['mb_fp32']:.1f} MB | fp16 {pricing['mb_fp16']:.1f} MB"
            f" | int8 {pricing['mb_int8']:.1f} MB | build {build_s:.1f}s"
            f" | int8 1-cos {pricing['roundtrip_vs_fp32']['int8']['mean_one_minus_cos']:.2e}"
        )

        # ---- scoring on the REAL component only (verdict evidence)
        pool = [tuple(p) for p in bench["real"]]
        full_row = rank_bench(full.normalized, full.word_to_idx, pool)
        variants = {}
        for name, mat in (("fp32", fp32_nrm), ("fp16", fp16_loaded), ("int8", int8_loaded)):
            pure = rank_bench(mat, full.word_to_idx, pool)
            hyb = hybrid_bench(mat, full.normalized, full.word_to_idx, pool, HYBRID_TOPK)
            paired = paired_hybrid_vs_full(hyb, full_row)
            variants[name] = {
                "pure_c_rank": {k: pure[k] for k in ("top1", "top5", "top20")},
                "hybrid": {k: hyb[k] for k in ("top1", "top5", "top20", "recall_at_slate", "pairs_evaluated")},
                "hybrid_vs_full_paired": paired,
            }
            print(
                f"  {name}: pure-C top5 {pure['top5']:.3f} | hybrid top1 {hyb['top1']:.3f}"
                f" top5 {hyb['top5']:.3f} (full {full_row['top1']:.3f}/{full_row['top5']:.3f})"
                f" | paired net top5 {paired['top5']['net_pp']:+.1f} pp"
                f" CI [{paired['top5']['ci95_pp'][0]:+.1f}, {paired['top5']['ci95_pp'][1]:+.1f}]"
            )
        # slate agreement vs the fp32 brute-force sweep
        hyb32 = hybrid_bench(fp32_nrm, full.normalized, full.word_to_idx, pool, HYBRID_TOPK)
        for name, mat in (("fp16", fp16_loaded), ("int8", int8_loaded)):
            hybq = hybrid_bench(mat, full.normalized, full.word_to_idx, pool, HYBRID_TOPK)
            a, b = hyb32["_in_slate"], hybq["_in_slate"]
            variants[name]["slate_agreement_vs_fp32"] = {
                "both_contain_correction": sum(1 for x, y in zip(a, b) if x and y),
                "fp32_only": sum(1 for x, y in zip(a, b) if x and not y),
                "variant_only": sum(1 for x, y in zip(a, b) if y and not x),
                "both_miss": sum(1 for x, y in zip(a, b) if not x and not y),
                "agreement_rate": float(np.mean([x == y for x, y in zip(a, b)])),
            }

        # ---- latency: batch-1 per-suggest on real queries
        lat_pool = pool[:LATENCY_QUERIES]
        samples = {"retrieval_fp32_amortized": [], "hybrid_fp32_amortized": [], "hybrid_int8_amortized": [], "hybrid_int8_lean": []}
        for typo, _c in lat_pool:
            t0 = time.perf_counter()
            q = enc.sess.run(["embedding"], {"char_ids": enc.ids_for([typo])})[0]
            qn = q[0] / max(float(np.linalg.norm(q[0])), 1e-9)
            sims = fp32_nrm @ qn
            np.argpartition(-sims, 20)[:20]
            samples["retrieval_fp32_amortized"].append((time.perf_counter() - t0) * 1e3)
        for typo, _c in lat_pool:
            t = full.word_to_idx.get(typo)
            if t is None:
                continue
            t0 = time.perf_counter()
            q = enc.sess.run(["embedding"], {"char_ids": enc.ids_for([typo])})[0]
            qn = q[0] / max(float(np.linalg.norm(q[0])), 1e-9)
            sims = fp32_nrm @ qn
            sims[t] = -np.inf
            slate = np.argpartition(-sims, HYBRID_TOPK - 1)[:HYBRID_TOPK]
            ft_sims = full.normalized[slate] @ full.normalized[t]
            np.argpartition(-ft_sims, 5)[:5]
            samples["hybrid_fp32_amortized"].append((time.perf_counter() - t0) * 1e3)
        for typo, _c in lat_pool:
            t = full.word_to_idx.get(typo)
            if t is None:
                continue
            t0 = time.perf_counter()
            q = enc.sess.run(["embedding"], {"char_ids": enc.ids_for([typo])})[0]
            qn = q[0] / max(float(np.linalg.norm(q[0])), 1e-9)
            sims = int8_loaded @ qn
            sims[t] = -np.inf
            slate = np.argpartition(-sims, HYBRID_TOPK - 1)[:HYBRID_TOPK]
            ft_sims = full.normalized[slate] @ full.normalized[t]
            np.argpartition(-ft_sims, 5)[:5]
            samples["hybrid_int8_amortized"].append((time.perf_counter() - t0) * 1e3)
        for typo, _c in lat_pool:
            t = full.word_to_idx.get(typo)
            if t is None:
                continue
            t0 = time.perf_counter()
            q = enc.sess.run(["embedding"], {"char_ids": enc.ids_for([typo])})[0]
            qn = q[0] / max(float(np.linalg.norm(q[0])), 1e-9)
            sims = int8_matvec_lean(qn, q8, s8)
            sims[t] = -np.inf
            slate = np.argpartition(-sims, HYBRID_TOPK - 1)[:HYBRID_TOPK]
            ft_sims = full.normalized[slate] @ full.normalized[t]
            np.argpartition(-ft_sims, 5)[:5]
            samples["hybrid_int8_lean"].append((time.perf_counter() - t0) * 1e3)

        def stats(vals: list[float]) -> dict:
            arr = np.asarray(vals)
            return {"n": len(arr), "median_ms": round(float(np.median(arr)), 2), "p90_ms": round(float(np.percentile(arr, 90)), 2)}

        latency = {k: stats(v) for k, v in samples.items()}
        latency["note"] = (
            "batch-1 per-suggest on real bench queries; amortized = matrix dequantized once at load "
            "(repo tier precedent); lean = int8 dequantized per query in 8k-row chunks, no fp32 matrix in RAM"
        )
        print(
            f"  latency ms (median/p90): fp32 {latency['hybrid_fp32_amortized']['median_ms']}/"
            f"{latency['hybrid_fp32_amortized']['p90_ms']} | int8 amortized"
            f" {latency['hybrid_int8_amortized']['median_ms']}/{latency['hybrid_int8_amortized']['p90_ms']}"
            f" | int8 lean {latency['hybrid_int8_lean']['median_ms']}/{latency['hybrid_int8_lean']['p90_ms']}"
        )

        langs_out[lang] = {
            "real_pairs": len(pool),
            "full_tier": {k: full_row[k] for k in ("top1", "top5", "top20", "pairs_evaluated", "pairs_typo_oov", "pairs_correction_oov")},
            "matrix_pricing": pricing,
            "variants": variants,
            "latency_ms_per_suggest": latency,
            "frozen": {"real_pairs": frozen["languages"][lang]["real_pairs"], "synth_pairs": frozen["languages"][lang]["synth_pairs"]},
        }
        del full, built, fp32_nrm, fp16_stored, fp16_loaded, q8, s8, int8_loaded

    # ---- verdict per the pre-declared rule (binding on the int8 configuration)
    model_int8_mb = manifest["c_typo_v2"]["int8_mb"]
    clause_a: dict[str, dict] = {}
    for lang in ("en", "de"):
        p = langs_out[lang]["variants"]["int8"]["hybrid_vs_full_paired"]["top5"]
        clause_a[lang] = {
            "net_pp": p["net_pp"],
            "ci95_pp": p["ci95_pp"],
            "pass": bool(p["net_pp"] >= CLEAR_WIN_PP and p["ci95_pp"][0] > 0.0),
        }
    clause_b: dict[str, dict] = {}
    for lang, data in langs_out.items():
        n = data["variants"]["int8"]["hybrid"]["pairs_evaluated"]
        if n < 20:
            continue
        p = data["variants"]["int8"]["hybrid_vs_full_paired"]["top5"]
        clause_b[lang] = {
            "n": n,
            "net_pp": p["net_pp"],
            "pass": bool(p["net_pp"] >= -DEGRADE_PP),
        }
    clause_c_size_mb = {
        "model_int8_mb": model_int8_mb,
        "matrix_int8_mb_per_language": {l: langs_out[l]["matrix_pricing"]["mb_int8"] for l in langs_out},
    }
    clause_c_size_mb["incremental_mb_en"] = model_int8_mb + langs_out["en"]["matrix_pricing"]["mb_int8"]
    clause_c_size_mb["incremental_mb_all_four"] = model_int8_mb + sum(langs_out[l]["matrix_pricing"]["mb_int8"] for l in langs_out)
    lat_med = langs_out["en"]["latency_ms_per_suggest"]["hybrid_int8_amortized"]["median_ms"]
    clause_c = {
        "size": clause_c_size_mb,
        "size_pass_en_only": bool(clause_c_size_mb["incremental_mb_en"] <= SIZE_CAP_MB),
        "size_pass_all_four": bool(clause_c_size_mb["incremental_mb_all_four"] <= SIZE_CAP_MB),
        "latency_median_ms": lat_med,
        "latency_pass": bool(lat_med <= LATENCY_CAP_MS),
    }
    verdict = {
        "rule": (
            "SHIP the hybrid (C v2 int8 + per-language int8 retrieval matrix) as an OPT-IN registry "
            f"resource only if (a) it beats the fastText FULL tier top-5 by >= {CLEAR_WIN_PP:g} pp "
            "(paired net, 95% CI excluding 0) on the en AND de real components, (b) it degrades "
            f"nowhere by more than {DEGRADE_PP:g} pp top-5 on any real component with n >= 20 pairs, "
            f"(c) the incremental artifact (model + per-language int8 matrix) <= {SIZE_CAP_MB:g} MB "
            f"and median per-suggest latency <= {LATENCY_CAP_MS:g} ms on the int8 path. Verdict binds "
            "on the int8 configuration (the only variant under the size cap); synth components are "
            "not verdict evidence (generator-domain). Honest reject is a complete outcome; no "
            "registry change, no tag on a reject."
        ),
        "clause_a_clear_win": clause_a,
        "clause_b_degrades_nowhere": clause_b,
        "clause_c_price": clause_c,
        "ship": bool(all(c["pass"] for c in clause_a.values()) and all(c["pass"] for c in clause_b.values()) and clause_c["size_pass_en_only"] and clause_c["latency_pass"]),
    }

    report = {
        "plan": "115 hybrid retrieval pricing",
        "hybrid": "C v2 top-20 retrieval + fastText FULL rescore (plan-114 survivor)",
        "decision_rule": verdict,
        "matrix_variants": (
            "fp32 = plan-114 brute-force sweep; fp16/int8 = stored-quantized matrices, "
            "dequantized and row-renormalized at load; int8 = symmetric per-row, one fp16 scale per row"
        ),
        "languages": langs_out,
        "models": {"c_v2": manifest["c_typo_v2"]},
        "hardware": "Apple M1 Max (arm64), CPUExecutionProvider",
        "report_only": True,
        "generated_at": iso_now(),
    }
    out = repo / "eval" / "reports" / "hybrid-pricing.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out.relative_to(repo)}")
    print(f"VERDICT: ship={verdict['ship']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
