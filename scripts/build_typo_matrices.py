#!/usr/bin/env python3
"""Build prebuilt KTM1 typo-retrieval matrices (plans 136/12/13).

Derives one matrix per language from the frozen typo bi-encoder + the
language's full-tier ONNX/vocab pair, writes the .ktm1 artifact and its
descriptor (models/{lang}/typo-matrix.json), then optionally regenerates
the registry. The export itself is the kotoshu-rs matrix_export example
(TypoIndex::write_rows); this script is the bulk orchestration so the
next language is one command, not a one-off session.

Usage:
  # Build de + it from local full-tier artifacts (already in the repo
  # or downloaded under --tiers-dir):
  python3 scripts/build_typo_matrices.py \\
      --langs de,it \\
      --typo-onnx ~/.cache/kotoshu/models/typo/typo.biencoder.onnx \\
      --typo-vocab ~/.cache/kotoshu/models/typo/typo.biencoder.vocab.json \\
      --tiers-dir /tmp/ktm1-build \\
      --matrix-export /path/to/matrix_export

  # After writing descriptors, regenerate the registry at a release tag:
  python3 scripts/build_registry.py --tag v1.7.0

The exporter binary is required (cargo build -p kotoshu --features model
--release --example matrix_export). Tier files default to
{tiers_dir}/fasttext.{lang}.onnx + {repo}/models/{lang}/fasttext.{lang}.vocab.json
(the vocab sidecars live in this repo; the 120 MB ONNX tiers usually
come from the media host).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = Path.home() / "src/kotoshu/kotoshu-rs/target/release/examples/matrix_export"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_full_sha(registry_path: Path, lang: str) -> str:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    for resource in data.get("resources", {}).values():
        if (
            resource.get("type") == "model"
            and resource.get("language") == lang
            and isinstance(resource.get("tier"), dict)
            and resource["tier"].get("name") == "full"
        ):
            return resource["sha256"]
    raise SystemExit(f"error: no full-tier entry for {lang} in {registry_path}")


def build_one(
    lang: str,
    *,
    exporter: Path,
    typo_onnx: Path,
    typo_vocab: Path,
    tier_onnx: Path,
    tier_vocab: Path,
    out_dir: Path,
    full_sha: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"typo.matrix.{lang}.ktm1"
    cmd = [
        str(exporter),
        str(typo_onnx),
        str(typo_vocab),
        str(tier_onnx),
        str(tier_vocab),
        str(artifact),
    ]
    print(f"== {lang}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        raise SystemExit(f"error: matrix_export failed for {lang} (rc={result.returncode})")
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)

    size = artifact.stat().st_size
    sha = sha256_of(artifact)
    magic = artifact.read_bytes()[:4]
    if magic != b"KTM1":
        raise SystemExit(f"error: {artifact} does not start with KTM1 (got {magic!r})")

    descriptor = {
        "plan": "13 prebuilt typo-retrieval matrix (KTM1 v1)",
        "kind": f"typo-retrieval matrix; rows index-parallel to the {lang} full-tier vocabulary",
        "vocab_size": 100_000,
        "dims": 256,
        "bytes": size,
        "sha256": sha,
        "paired_vocab": f"kotoshu://models/{lang}/full @ sha256 {full_sha}",
        "built_with": "kotoshu-rs examples/matrix_export (TypoIndex::write_rows)",
        "license": "CC-BY-SA-3.0",
        "min_engine_version": "1.1",
        "release_tag": None,
        "eval_ref": "eval/reports/hybrid-pricing.json",
    }
    desc_path = out_dir / "typo-matrix.json"
    desc_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    print(f"   {lang}: {size} bytes sha={sha[:16]}… → {artifact}", flush=True)
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", required=True, help="comma-separated ISO 639-1 codes")
    ap.add_argument("--typo-onnx", required=True, type=Path, help="path to typo.biencoder.onnx")
    ap.add_argument("--typo-vocab", required=True, type=Path, help="path to typo.biencoder.vocab.json")
    ap.add_argument(
        "--tiers-dir",
        type=Path,
        default=None,
        help="directory holding fasttext.{lang}.onnx (default: look under models/{lang}/)",
    )
    ap.add_argument(
        "--matrix-export",
        type=Path,
        default=DEFAULT_EXPORT,
        help="path to the matrix_export binary (default: ~/src/kotoshu/kotoshu-rs/target/release/examples/matrix_export)",
    )
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument(
        "--install",
        action="store_true",
        help="copy artifacts + descriptors into models/{lang}/ (default: write next to the tier)",
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    langs = [part.strip() for part in args.langs.split(",") if part.strip()]
    if not langs:
        sys.exit("error: --langs is empty")
    for path, label in (
        (args.matrix_export, "matrix_export binary"),
        (args.typo_onnx, "typo onnx"),
        (args.typo_vocab, "typo vocab"),
    ):
        if not path.is_file():
            sys.exit(f"error: {label} not found: {path}")

    registry_path = root / "registry.json"
    if not registry_path.is_file():
        sys.exit(f"error: {registry_path} missing (needed for paired_vocab sha)")

    built = []
    for lang in langs:
        if args.tiers_dir is not None:
            tier_onnx = args.tiers_dir / f"fasttext.{lang}.onnx"
        else:
            tier_onnx = root / "models" / lang / f"fasttext.{lang}.onnx"
        tier_vocab = root / "models" / lang / f"fasttext.{lang}.vocab.json"
        for path, label in ((tier_onnx, "tier onnx"), (tier_vocab, "tier vocab")):
            if not path.is_file():
                sys.exit(f"error: {label} for {lang} not found: {path}")

        out_dir = (root / "models" / lang) if args.install else (tier_onnx.parent if args.tiers_dir else root / "models" / lang)
        if args.tiers_dir is not None and not args.install:
            out_dir = args.tiers_dir
        full_sha = load_full_sha(registry_path, lang)
        artifact = build_one(
            lang,
            exporter=args.matrix_export,
            typo_onnx=args.typo_onnx,
            typo_vocab=args.typo_vocab,
            tier_onnx=tier_onnx,
            tier_vocab=tier_vocab,
            out_dir=out_dir,
            full_sha=full_sha,
        )
        if args.install and out_dir != root / "models" / lang:
            dest = root / "models" / lang
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, dest / artifact.name)
            shutil.copy2(out_dir / "typo-matrix.json", dest / "typo-matrix.json")
            print(f"   installed into {dest}", flush=True)
        built.append(lang)

    print(f"built {len(built)} matrix(es): {', '.join(built)}")
    print("next: python3 scripts/build_registry.py --tag <release-tag>  # then validate + PR")


if __name__ == "__main__":
    main()
