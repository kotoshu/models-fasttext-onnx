#!/usr/bin/env python3
"""Build smaller tiers (fluency, mini) from full fastText ONNX models.

flueuncy: full 300 dims kept, int8 per row, vocab truncated to the most
frequent 50k rows (~15 MB). Empirically the SVD-dim-reduction recipes
(128-160 dims) lose far too much neighbour-ranking fidelity (rank_corr
~0.87-0.91 vs the 0.97 gate); int8 without dim reduction preserves
cosine rankings almost exactly, so the size saving comes from the vocab
cut instead. Eval reports in eval/reports/ document this.
mini: full dims with a 10k vocab (~3 MB), falling back to dim-reduced
variants if gates require.

Dimensionality reduction (when used) is truncated SVD computed as the
eigendecomposition of the 300x300 uncentered covariance X.T @ X
(fastText embeddings are meant to pass through the origin, so no
centering), projecting X @ V[:dims].T. np.linalg.svd is never called on
the tall matrix directly.

For each tier the script writes the tier .onnx + .vocab.json, runs the eval
harness (eval/run_eval.py) and writes eval/reports/{lang}.{tier}.json. If a
gate fails it climbs a small recipe ladder within the allowed budget and
records every attempt in the final report; gates are never weakened.
Exit status is nonzero if any gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

QUANT_MAX_ABS_TOL = 0.05

FLUENCY_LADDER = [  # (dims, vocab_size)
    (300, 50_000),
    (300, 60_000),
    (300, 100_000),
]
MINI_LADDER = [  # (dims, vocab_size)
    (300, 10_000),
    (256, 12_000),
    (160, 20_000),
    (64, 30_000),
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def manifest_languages(repo: Path) -> list[str]:
    manifest = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))
    langs = sorted({entry["language"] for entry in manifest["resources"].values() if entry.get("type") == "onnx"})
    return langs


def full_model_sha256(repo: Path, lang: str) -> str:
    manifest = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))
    rel = f"models/{lang}/fasttext.{lang}.onnx"
    entry = manifest["resources"][rel]
    actual = sha256((repo / rel).read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        raise RuntimeError(f"{rel}: sha256 mismatch vs manifest.json — refusing to build from it")
    return entry["sha256"]


def reduce_dims_uncentered(x: np.ndarray, dims: int) -> np.ndarray:
    if dims >= x.shape[1]:
        return x
    cov = (x.T @ x).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1][:dims]
    return x @ eigenvectors[:, order].astype(np.float32)


def quantize_int8_per_row(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = (np.abs(y).max(axis=1) / 127.0).astype(np.float32)
    scale[scale == 0.0] = 1.0
    q = np.rint(y / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def make_tier_model(q: np.ndarray, scale: np.ndarray, tier: str) -> onnx.ModelProto:
    vocab_size, dims = q.shape

    input_tensor = helper.make_tensor_value_info("word_index", TensorProto.INT64, [1])
    output_tensor = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [dims])

    nodes = [
        helper.make_node(
            "Constant", [], ["q_embeddings"], value=numpy_helper.from_array(q, name="q_embeddings")
        ),
        helper.make_node(
            "Constant", [], ["row_scale"], value=numpy_helper.from_array(scale, name="row_scale")
        ),
        helper.make_node(
            "Constant",
            [],
            ["scale_shape"],
            value=numpy_helper.from_array(np.array([1, 1], dtype=np.int64), name="scale_shape"),
        ),
        helper.make_node("Gather", ["q_embeddings", "word_index"], ["emb_i8"], axis=0),
        helper.make_node("Gather", ["row_scale", "word_index"], ["row_scale_i"], axis=0),
        helper.make_node("Reshape", ["row_scale_i", "scale_shape"], ["s_1x1"]),
        helper.make_node("Cast", ["emb_i8"], ["emb_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["emb_f", "s_1x1"], ["embedding_flat"]),
        helper.make_node("Squeeze", ["embedding_flat"], ["embedding"], axes=[0]),
    ]
    graph = helper.make_graph(nodes, f"fasttext_{tier}_embedding", [input_tensor], [output_tensor])
    model = helper.make_model(
        graph,
        producer_name="kotoshu-fasttext-converter",
        producer_version="1.0.0",
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=11,
    )
    from onnx import StringStringEntryProto

    model.metadata_props.append(StringStringEntryProto(key="vocabulary_size", value=str(vocab_size)))
    model.metadata_props.append(StringStringEntryProto(key="embedding_dimension", value=str(dims)))
    model.metadata_props.append(StringStringEntryProto(key="model_type", value="fasttext_embedding"))
    model.metadata_props.append(StringStringEntryProto(key="quantization", value="int8-per-row"))
    model.metadata_props.append(StringStringEntryProto(key="tier", value=tier))
    return model


def tier_vocab_map(full_word_to_idx: dict, vocab_size: int | None) -> dict:
    if vocab_size is None:
        return dict(full_word_to_idx)
    words = sorted(full_word_to_idx.items(), key=lambda kv: kv[1])[:vocab_size]
    return {w: i for i, (w, _) in enumerate(words)}


def write_vocab_json(path: Path, word_to_idx: dict) -> None:
    data = {"vocab_size": len(word_to_idx), "word_to_idx": word_to_idx}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_tier_onnx(onnx_path: Path, dequant: np.ndarray, projected: np.ndarray) -> dict:
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    idxs = sorted({0, 1, dequant.shape[0] // 2, dequant.shape[0] - 1})
    worst_runtime = 0.0
    for i in idxs:
        out = session.run(["embedding"], {"word_index": np.array([i], dtype=np.int64)})[0]
        worst_runtime = max(worst_runtime, float(np.abs(out - dequant[i]).max()))
    quant_max_abs = float(np.abs(projected - dequant).max())
    check = {
        "runtime_vs_dequant_max_abs": worst_runtime,
        "dequant_vs_projected_max_abs": quant_max_abs,
        "dequant_vs_projected_tol": QUANT_MAX_ABS_TOL,
        "passed": quant_max_abs < QUANT_MAX_ABS_TOL and worst_runtime < 1e-4,
    }
    if not check["passed"]:
        raise RuntimeError(f"{onnx_path}: quantization check failed: {check}")
    return check


def build_tier_artifacts(
    repo: Path,
    lang: str,
    tier: str,
    x: np.ndarray,
    full_word_to_idx: dict,
    dims: int,
    vocab_size: int | None,
) -> dict:
    sub = x if vocab_size is None else x[:vocab_size]
    projected = reduce_dims_uncentered(sub, dims)
    q, scale = quantize_int8_per_row(projected)
    dequant = q.astype(np.float32) * scale[:, None]

    onnx_path = repo / "models" / lang / f"fasttext.{lang}.{tier}.onnx"
    vocab_path = repo / "models" / lang / f"fasttext.{lang}.{tier}.vocab.json"
    onnx.save(make_tier_model(q, scale, tier), str(onnx_path))
    write_vocab_json(vocab_path, tier_vocab_map(full_word_to_idx, vocab_size))

    check = verify_tier_onnx(onnx_path, dequant, projected)
    return {
        "onnx_path": onnx_path,
        "vocab_path": vocab_path,
        "dims": dims,
        "vocab_size": dequant.shape[0],
        "check": check,
    }


def write_tiers_entry(repo: Path, lang: str, tier: str, info: dict, source_sha: str) -> None:
    entry = {
        "dims": info["dims"],
        "vocab_size": info["vocab_size"],
        "quantization": "int8-per-row",
        "bytes": info["onnx_path"].stat().st_size,
        "sha256": sha256(info["onnx_path"].read_bytes()).hexdigest(),
        "vocab_sha256": sha256(info["vocab_path"].read_bytes()).hexdigest(),
        "vocab_bytes": info["vocab_path"].stat().st_size,
        "eval_ref": f"eval/reports/{lang}.{tier}.json",
        "source_full_sha256": source_sha,
    }
    tiers_path = repo / "models" / lang / "tiers.json"
    data = {}
    if tiers_path.exists():
        data = json.loads(tiers_path.read_text(encoding="utf-8"))
    data["language"] = lang
    data.setdefault("tiers", {})[tier] = entry
    data["generated_at"] = iso_now()
    tiers_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def metric_summary(report: dict) -> dict:
    m = report["metrics"]
    return {
        "rank_corr": round(m["rank_corr"]["mean_spearman"], 6),
        "top1_agreement": round(m["top1_agreement"]["agreement"], 6),
        "recipe": report["recipe"],
    }


def run_ladder(repo: Path, run_eval, lang: str, tier: str, ladder, x, word_to_idx) -> tuple[dict | None, dict, list]:
    attempts: list[dict] = []
    info: dict | None = None
    report: dict | None = None
    for recipe in ladder:
        dims = recipe[0]
        vocab_size = recipe[1] if len(recipe) > 1 else None
        info = build_tier_artifacts(repo, lang, tier, x, word_to_idx, dims, vocab_size)
        report = run_eval.evaluate(
            repo, lang, tier, build_check=info["check"], attempts=list(attempts), write=False
        )
        passed = report["gates"]["passed"]
        attempts.append({"recipe": {"dims": dims, "vocab_size": info["vocab_size"]}, "passed": passed, "metrics": metric_summary(report)})
        print(
            f"  {lang}/{tier} dims={dims} vocab={info['vocab_size']}: "
            f"rank_corr={report['metrics']['rank_corr']['mean_spearman']:.4f} "
            f"top1={report['metrics']['top1_agreement']['agreement']:.4f} "
            f"{'PASS' if passed else 'fail'}"
        )
        if passed:
            return info, report, attempts
    return None, report, attempts


def build_language(repo: Path, run_eval, lang: str) -> list[str]:
    model_dir = repo / "models" / lang
    full = onnx.load(str(model_dir / f"fasttext.{lang}.onnx"))
    x = np.ascontiguousarray(constant_array(full, "word_embeddings"), dtype=np.float32)
    vocab = json.loads((model_dir / f"fasttext.{lang}.vocab.json").read_text(encoding="utf-8"))
    if x.shape != (vocab["vocab_size"], x.shape[1]) or len(vocab["word_to_idx"]) != x.shape[0]:
        raise ValueError(f"{lang}: vocab json does not match embedding matrix {x.shape}")
    source_sha = full_model_sha256(repo, lang)

    failures: list[str] = []
    for tier, ladder in (("fluency", FLUENCY_LADDER), ("mini", MINI_LADDER)):
        passed_info, report, attempts = run_ladder(repo, run_eval, lang, tier, ladder, x, vocab["word_to_idx"])
        report["attempts"] = attempts
        reports_dir = repo / "eval" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / f"{lang}.{tier}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if passed_info is not None:
            write_tiers_entry(repo, lang, tier, passed_info, source_sha)
        else:
            failures.append(f"{lang}/{tier}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fluency/mini tiers from full fastText ONNX models")
    parser.add_argument("--lang", nargs="+", help="language codes, e.g. en de")
    parser.add_argument("--all", action="store_true", help="build every language in manifest.json")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    known = manifest_languages(repo)
    if args.all:
        langs = known
    elif args.lang:
        unknown = [l for l in args.lang if l not in known]
        if unknown:
            parser.error(f"unknown languages {unknown}; manifest has {known}")
        langs = args.lang
    else:
        parser.error("specify --lang or --all")

    sys.path.insert(0, str(repo / "eval"))
    import run_eval  # noqa: E402

    all_failures: list[str] = []
    for lang in langs:
        print(f"[{lang}] loading full model")
        all_failures.extend(build_language(repo, run_eval, lang))

    if all_failures:
        print(f"GATE FAILURES: {', '.join(all_failures)}", file=sys.stderr)
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
