#!/usr/bin/env python3
"""Promote the plan-114/115 typo bi-encoder candidate to models/typo/.

Copies the c_typo_v2 int8 artifact (eval/candidates/c_typo_v2, produced by
scripts/train_typo_biencoder_v2.py and priced by
eval/hybrid_pricing_bench.py) into the opt-in registry resource pair
models/typo/typo.biencoder.onnx + typo.biencoder.vocab.json, and writes
the models/typo/typo.json descriptor (registry ground truth, validate
side) plus models/typo/metadata.json (manifest facts). Idempotent: same
inputs give same outputs modulo generated_at; the artifacts are copied
verbatim.

The retrieval side is NOT shipped as a download: the per-language
100k x 256 vocabulary matrix is derived at load (encode the vocab through
this model, ~5 s per language on an M1 Max, priced in
eval/reports/hybrid-pricing.{json,md}: 25.8 MB stored int8 / 102.4 MB
fp32 in RAM, or the RAM-lean int8 path at ~27 ms per suggest).

Usage:
  python3 scripts/promote_typo_biencoder.py --repo-root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx

CANDIDATE = "c_typo_v2"
OUT_DIR = "models/typo"
ONNX_NAME = "typo.biencoder.onnx"
VOCAB_NAME = "typo.biencoder.vocab.json"
EVAL_REF = "eval/reports/hybrid-pricing.json"
CORPUS_URL = "https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz"
LICENSE = "CC-BY-SA-3.0"  # vocab conditioning derives from the CC-BY-SA-3.0 fastText vectors
MIN_ENGINE_VERSION = "0.7"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()

    cand_dir = repo / "eval" / "candidates" / CANDIDATE
    manifest_path = repo / "eval" / "candidates" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = manifest[CANDIDATE]

    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / ONNX_NAME
    vocab_path = out / VOCAB_NAME
    shutil.copyfile(cand_dir / "model.int8.onnx", onnx_path)
    shutil.copyfile(cand_dir / "char_to_idx.json", vocab_path)

    opset = next(o.version for o in onnx.load(str(onnx_path)).opset_import if (o.domain or "ai.onnx") == "ai.onnx")
    char_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    descriptor = {
        "plan": "114 trained it, 115 priced and shipped it (opt-in)",
        "resource": "kotoshu://models/typo/typo-biencoder",
        "kind": "typo bi-encoder (hybrid retrieval half)",
        "architecture": receipt["architecture"],
        "dims": 256,
        "char_vocab_size": len(char_vocab),
        "quantization": "int8-dynamic",
        "bytes": onnx_path.stat().st_size,
        "sha256": file_sha256(onnx_path),
        "vocab_sha256": file_sha256(vocab_path),
        "vocab_bytes": vocab_path.stat().st_size,
        "training": receipt["training"],
        "data": receipt["data"],
        "determinism": receipt["determinism"],
        "retrieval": {
            "note": "derived at load, never downloaded: encode the language vocab through this model",
            "matrix": "per-language 100k x 256, row-L2-normalized",
            "variants_bytes": {"fp32": 102_400_000, "fp16": 51_200_000, "int8_incl_fp16_scales": 25_800_000},
            "build_s_per_100k_words": "4.8-5.7 (Apple M1 Max)",
            "int8_ranking": "identical hit@1/5/20 to the fp32 brute-force sweep on every frozen C-benchmark real component (plan 115)",
            "usage": "top-20 slate by cosine to the typo, rescored by the fastText FULL tier (the plan-114/115 hybrid)",
        },
        "verdict": (
            "plan-115 pre-declared rule, all clauses pass on real-pair components only: "
            "en +6.3 pp top-5 (CI +4.9/+7.7), de +28.6 pp (+7.1/+50.0, n=14), "
            "ru +11.1 pp (n=27), es +10.0 pp (n=20); no component regresses; "
            "0.481 MB artifact; 3.3-4.3 ms median per-suggest (int8 amortized)"
        ),
        "release_tag": None,
        "release_note": (
            "media-host mirror only (plan-113 additive template); primary stays null until the "
            "owner cuts a release carrying the typo assets, then this field and the registry "
            "entry flip together"
        ),
        "eval_ref": EVAL_REF,
        "source": {
            "trained_in_repo": "scripts/train_typo_biencoder_v2.py",
            "corpus": CORPUS_URL,
            "corpus_sha256": json.loads((repo / "eval" / "reports" / "cbench.frozen.json").read_text(encoding="utf-8"))["corpus_sha256"],
            "vocab_conditioning": "fastText Common Crawl vocabularies (models/<lang>/fasttext.<lang>.vocab.json)",
        },
        "license": LICENSE,
        "min_engine_version": MIN_ENGINE_VERSION,
        "generated_at": iso_now(),
    }
    (out / "typo.json").write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "version": descriptor["generated_at"],
        "language": "typo",
        "type": "onnx",
        "file": ONNX_NAME,
        "source_model": "char-BiGRU typo bi-encoder trained in-repo on the GitHub Typo Corpus (plan 114)",
        "conversion_method": "promote_typo_biencoder.py (from scripts/train_typo_biencoder_v2.py)",
        "opset_version": opset,
        "upstream": CORPUS_URL,
        "upstream_sha256": descriptor["source"]["corpus_sha256"],
        "license": LICENSE,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # keep the candidate-manifest audit trail pointing at the promotion
    if "promoted_to" not in receipt:
        receipt["promoted_to"] = {
            "resource": descriptor["resource"],
            "descriptor": "models/typo/typo.json",
            "plan": "115",
            "sha256": descriptor["sha256"],
            "vocab_sha256": descriptor["vocab_sha256"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"promoted {CANDIDATE} -> {OUT_DIR}/ ({descriptor['bytes']} B onnx, {descriptor['vocab_bytes']} B char vocab, opset {opset})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
