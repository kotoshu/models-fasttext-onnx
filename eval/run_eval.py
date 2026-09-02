#!/usr/bin/env python3
"""Eval harness for tiered fastText ONNX models.

Measures how well a derived tier model (fluency/mini) preserves the ranking
behaviour of its full-model parent and gates release on the measured numbers.

Metrics:
- rank_corr: mean Spearman correlation between tier and full cosine-score
  vectors, computed per probe word over identical comparison-word sets.
  Comparison words the tier cannot embed are dropped from BOTH score vectors
  so the comparison is apples-to-apples.
- top1_agreement: fraction of typo-rerank probes where tier and full models
  pick the same top candidate. This is a PROXY for the gem's real rerank
  pipeline (see METRIC_NOTE).
- coverage: frequency-weighted vocabulary coverage (weight 1/(rank+1)).
  Reported, not gated.

Determinism: every random draw comes from a seeded PCG64 stream,
np.random.default_rng([42, crc32(language)]), so a given (language, tier)
model pair always yields the same report apart from timestamps.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from scipy.stats import spearmanr

SEED = 42

RANK_CORR_PROBES = 2000
RANK_CORR_BUCKETS = 20
RANK_CORR_COMPARISONS = 3000
RANK_CORR_LO = 50
RANK_CORR_HI = 30_000  # exclusive

TOP1_PROBES = 400
TOP1_LO = 1_000
TOP1_HI = 30_000  # exclusive; also keeps probes inside the mini vocab
TOP1_NEIGHBORS = 20
TOP1_WINDOW = 20_000
TYPO_ATTEMPTS = 50

ORT_SPOT_CHECK_TOL = 1e-4

METRIC_NOTE = (
    "top1_agreement is a proxy for the gem's spelling-correction rerank "
    "pipeline, not the pipeline itself: probes are edit-distance-1 typos "
    "that are themselves in-vocabulary, the candidate set is the true word "
    "plus the top-20 nearest neighbours of the typo under the full model "
    "over a 20000-word sample window, and each model scores candidates by "
    "cosine to the typo. It will be replaced by gem conformance vectors "
    "(kotoshu-rs M3)."
)


@dataclass
class LoadedModel:
    tier: str
    embeddings: np.ndarray  # fp32 [V, d]; tier models are dequantized rows
    normalized: np.ndarray  # row-L2-normalized copy
    word_to_idx: dict
    dims: int
    vocab_size: int
    quantization: str | None


def constant_array(model: onnx.ModelProto, name: str) -> np.ndarray:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            # match either the Constant node's output name or its value
            # tensor's own name (the full models name the value tensor only)
            if attr.name == "value" and (attr.t.name == name or name in node.output):
                return numpy_helper.to_array(attr.t)
    raise KeyError(f"array {name!r} not found in model")


def metadata_dict(model: onnx.ModelProto) -> dict:
    return {p.key: p.value for p in model.metadata_props}


def _normalized(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _check_consistency(emb: np.ndarray, vocab: dict, meta: dict, path: str) -> None:
    rows, dims = emb.shape
    problems = []
    if vocab["vocab_size"] != rows or len(vocab["word_to_idx"]) != rows:
        problems.append(f"vocab json size {vocab['vocab_size']}/{len(vocab['word_to_idx'])} != matrix rows {rows}")
    if "vocabulary_size" in meta and int(meta["vocabulary_size"]) != rows:
        problems.append(f"metadata vocabulary_size {meta['vocabulary_size']} != {rows}")
    if "embedding_dimension" in meta and int(meta["embedding_dimension"]) != dims:
        problems.append(f"metadata embedding_dimension {meta['embedding_dimension']} != {dims}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))


def load_full_model(onnx_path: Path, vocab_path: Path) -> LoadedModel:
    model = onnx.load(str(onnx_path))
    emb = np.ascontiguousarray(constant_array(model, "word_embeddings"), dtype=np.float32)
    meta = metadata_dict(model)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    _check_consistency(emb, vocab, meta, str(onnx_path))
    return LoadedModel(
        tier="full",
        embeddings=emb,
        normalized=_normalized(emb),
        word_to_idx=vocab["word_to_idx"],
        dims=emb.shape[1],
        vocab_size=emb.shape[0],
        quantization=None,
    )


def load_tier_model(onnx_path: Path, vocab_path: Path) -> LoadedModel:
    model = onnx.load(str(onnx_path))
    q = constant_array(model, "q_embeddings")
    scale = constant_array(model, "row_scale")
    if q.dtype != np.int8 or scale.dtype != np.float32:
        raise ValueError(f"{onnx_path}: expected int8 q_embeddings / fp32 row_scale")
    if q.shape[0] != scale.shape[0]:
        raise ValueError(f"{onnx_path}: q_embeddings rows {q.shape[0]} != row_scale rows {scale.shape[0]}")
    dequant = q.astype(np.float32) * scale[:, None]
    meta = metadata_dict(model)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    _check_consistency(dequant, vocab, meta, str(onnx_path))
    spot_check_against_runtime(onnx_path, dequant)
    return LoadedModel(
        tier=meta.get("tier", "unknown"),
        embeddings=dequant,
        normalized=_normalized(dequant),
        word_to_idx=vocab["word_to_idx"],
        dims=dequant.shape[1],
        vocab_size=dequant.shape[0],
        quantization=meta.get("quantization"),
    )


def spot_check_against_runtime(onnx_path: Path, dequant: np.ndarray, tol: float = ORT_SPOT_CHECK_TOL) -> float:
    """Confirm the dequantized rows match what onnxruntime actually outputs."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    n = dequant.shape[0]
    idxs = sorted({0, 1, n // 2, n - 1} | ({29_999} if n > 30_000 else set()))
    worst = 0.0
    for i in idxs:
        out = session.run(["embedding"], {"word_index": np.array([i], dtype=np.int64)})[0]
        worst = max(worst, float(np.abs(out - dequant[i]).max()))
    if worst > tol:
        raise RuntimeError(f"{onnx_path}: onnxruntime diverges from dequantized rows (max abs {worst:.3g} > {tol})")
    return worst


def make_rng(lang: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(lang.encode("utf-8"))])


def _bucketed_probe_ranks(rng: np.random.Generator, lo: int, hi: int, n: int, buckets: int) -> np.ndarray:
    per = n // buckets
    width = (hi - lo) // buckets
    ranks: list[int] = []
    for b in range(buckets):
        blo = lo + b * width
        bhi = lo + (b + 1) * width if b < buckets - 1 else hi
        ranks.extend(rng.integers(blo, bhi, size=per))
    return np.asarray(ranks, dtype=np.int64)


def rank_corr_metric(full: LoadedModel, tier: LoadedModel, rng: np.random.Generator) -> dict:
    hi = min(RANK_CORR_HI, tier.vocab_size)
    probes = _bucketed_probe_ranks(rng, RANK_CORR_LO, hi, RANK_CORR_PROBES, RANK_CORR_BUCKETS)
    rhos: list[float] = []
    comps_seen: list[int] = []
    skipped = 0
    for r in probes:
        comps = rng.integers(0, full.vocab_size, size=RANK_CORR_COMPARISONS)
        comps = comps[comps < tier.vocab_size]  # scoreable by both models
        if comps.size < 2:
            skipped += 1
            continue
        sf = full.normalized[comps] @ full.normalized[r]
        st = tier.normalized[comps] @ tier.normalized[r]
        if np.ptp(sf) == 0.0 or np.ptp(st) == 0.0:
            skipped += 1
            continue
        rho = spearmanr(sf, st).statistic
        if np.isfinite(rho):
            rhos.append(float(rho))
            comps_seen.append(int(comps.size))
        else:
            skipped += 1
    return {
        "mean_spearman": float(np.mean(rhos)),
        "n_probes_scored": len(rhos),
        "n_probes_skipped": skipped,
        "mean_comparisons_per_probe": float(np.mean(comps_seen)),
        "rank_range": [RANK_CORR_LO, hi],
    }


def make_typo(rng: np.random.Generator, word: str) -> str | None:
    letters = list(string.ascii_lowercase)
    for _ in range(TYPO_ATTEMPTS):
        ops = ["insert", "substitute"]
        if len(word) >= 2:
            ops.append("delete")
        op = ops[int(rng.integers(len(ops)))]
        if op == "delete":
            pos = int(rng.integers(len(word)))
            typo = word[:pos] + word[pos + 1 :]
        elif op == "insert":
            pos = int(rng.integers(len(word) + 1))
            typo = word[:pos] + str(rng.choice(letters)) + word[pos:]
        else:
            pos = int(rng.integers(len(word)))
            c = str(rng.choice(letters))
            if c == word[pos]:
                continue
            typo = word[:pos] + c + word[pos + 1 :]
        if typo != word:
            return typo
    return None


def top1_agreement_metric(full: LoadedModel, tier: LoadedModel, rng: np.random.Generator) -> dict:
    hi = min(TOP1_HI, tier.vocab_size)
    probes = rng.integers(TOP1_LO, hi, size=TOP1_PROBES)
    rank_to_word = [None] * full.vocab_size
    for w, i in full.word_to_idx.items():
        rank_to_word[i] = w

    agreements = 0
    scored = 0
    skipped_typo = 0
    skipped_candidates = 0
    dropped_candidates = 0

    for r in probes:
        word = rank_to_word[int(r)]
        typo = make_typo(rng, word)
        if typo is None:
            skipped_typo += 1
            continue
        t_rank = full.word_to_idx.get(typo)
        if t_rank is None or typo not in tier.word_to_idx:
            # both models must be able to embed the typo for a paired score
            skipped_typo += 1
            continue

        window = rng.choice(full.vocab_size, size=TOP1_WINDOW, replace=False)
        sims = full.normalized[window] @ full.normalized[t_rank]
        k = min(TOP1_NEIGHBORS + 1, window.size)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        neighbours = []
        for oi in top:
            wr = int(window[oi])
            if wr != t_rank:
                neighbours.append(wr)
            if len(neighbours) == TOP1_NEIGHBORS:
                break

        seen: set[int] = set()
        cand_list: list[int] = []
        for c in [int(r)] + neighbours:
            if c not in seen:
                seen.add(c)
                cand_list.append(c)
        kept = [c for c in cand_list if c < tier.vocab_size]
        dropped_candidates += len(cand_list) - len(kept)
        cands = np.asarray(kept, dtype=np.int64)
        if cands.size < 2:
            skipped_candidates += 1
            continue

        sf = full.normalized[cands] @ full.normalized[t_rank]
        st = tier.normalized[cands] @ tier.normalized[t_rank]
        if int(cands[int(np.argmax(sf))]) == int(cands[int(np.argmax(st))]):
            agreements += 1
        scored += 1

    return {
        "agreement": (agreements / scored) if scored else 0.0,
        "n_probes_scored": scored,
        "n_probes_skipped_typo": skipped_typo,
        "n_probes_skipped_candidates": skipped_candidates,
        "mean_candidates_dropped_out_of_vocab": dropped_candidates / TOP1_PROBES,
        "rank_range": [TOP1_LO, hi],
    }


def coverage_metric(full_vocab_size: int, tier_vocab_size: int) -> dict:
    weights = 1.0 / (np.arange(full_vocab_size, dtype=np.float64) + 1.0)
    return {
        "frequency_weighted_coverage": float(weights[:tier_vocab_size].sum() / weights.sum()),
        "full_vocab_size": full_vocab_size,
        "tier_vocab_size": tier_vocab_size,
        "weighting": "1/(rank+1)",
    }


def load_gates(repo_root: Path) -> dict:
    return json.loads((repo_root / "eval" / "gates.json").read_text(encoding="utf-8"))


def apply_gates(metrics: dict, gates: dict) -> dict:
    checks = {
        "rank_corr": metrics["rank_corr"]["mean_spearman"] >= gates["rank_corr_min"],
        "top1_agreement": metrics["top1_agreement"]["agreement"] >= gates["top1_agreement_min"],
    }
    return {"thresholds": gates, "checks": checks, "passed": all(checks.values())}


def _file_info(repo_root: Path, rel: str) -> dict:
    p = repo_root / rel
    return {
        "path": rel,
        "bytes": p.stat().st_size,
        "sha256": sha256(p.read_bytes()).hexdigest(),
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def evaluate(
    repo_root: Path,
    lang: str,
    tier: str,
    *,
    build_check: dict | None = None,
    attempts: list | None = None,
    write: bool = True,
) -> dict:
    repo = Path(repo_root)
    model_dir = repo / "models" / lang
    full = load_full_model(model_dir / f"fasttext.{lang}.onnx", model_dir / f"fasttext.{lang}.vocab.json")
    tm = load_tier_model(model_dir / f"fasttext.{lang}.{tier}.onnx", model_dir / f"fasttext.{lang}.{tier}.vocab.json")

    rng = make_rng(lang)
    metrics = {
        "rank_corr": rank_corr_metric(full, tm, rng),
        "top1_agreement": top1_agreement_metric(full, tm, rng),
        "coverage": coverage_metric(full.vocab_size, tm.vocab_size),
    }
    gates = apply_gates(metrics, load_gates(repo)[tier])

    report = {
        "language": lang,
        "tier": tier,
        "recipe": {
            "dims": tm.dims,
            "vocab_size": tm.vocab_size,
            "quantization": tm.quantization,
            "reduction": "uncentered-covariance-eigendecomposition",
        },
        "metrics": metrics,
        "gates": gates,
        "metric_note": METRIC_NOTE,
        "determinism": {"seed": SEED, "rng": "np.random.default_rng([seed, crc32(language)])"},
        "files": {
            "tier_onnx": _file_info(repo, f"models/{lang}/fasttext.{lang}.{tier}.onnx"),
            "tier_vocab": _file_info(repo, f"models/{lang}/fasttext.{lang}.{tier}.vocab.json"),
            "full_onnx": _file_info(repo, f"models/{lang}/fasttext.{lang}.onnx"),
        },
        "generated_at": _iso_now(),
    }
    if build_check is not None:
        report["quantization_check"] = build_check
    if attempts is not None:
        report["attempts"] = attempts

    if write:
        reports_dir = repo / "eval" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / f"{lang}.{tier}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one tier model against its full-model parent")
    parser.add_argument("--lang", required=True, help="language code, e.g. en")
    parser.add_argument("--tier", required=True, choices=("fluency", "mini"))
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    args = parser.parse_args()

    report = evaluate(Path(args.repo_root), args.lang, args.tier)
    m = report["metrics"]
    print(
        f"{args.lang}/{args.tier}: rank_corr={m['rank_corr']['mean_spearman']:.4f} "
        f"top1={m['top1_agreement']['agreement']:.4f} "
        f"coverage={m['coverage']['frequency_weighted_coverage']:.4f} "
        f"passed={report['gates']['passed']}"
    )
    return 0 if report["gates"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
