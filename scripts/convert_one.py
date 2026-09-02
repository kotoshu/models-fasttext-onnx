#!/usr/bin/env python3
"""Convert one upstream fastText .vec file to ONNX for a single language.

Wraps scripts/fasttext_to_onnx.py (imported, not duplicated): parses the
.vec, writes fasttext.{lang}.onnx and fasttext.{lang}.vocab.json into
models/{lang}/ (or --out-dir), and updates metadata.json provenance
(source url/sha256, exact command, tool versions). Never touches
manifest.json or registry.json; regenerate those explicitly.

Usage:
    python scripts/convert_one.py --language en --vec cc.en.300.vec
    python scripts/convert_one.py --language toy --vec toy.vec --out-dir /tmp/toy
"""

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy  # noqa: E402
import onnx  # noqa: E402
import fasttext_to_onnx  # noqa: E402

try:
    import onnxruntime
except ImportError:  # version recording only; conversion does not need it
    onnxruntime = None

PIN_FILE = SCRIPTS_DIR / "upstream_versions.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_url_for(language):
    if not PIN_FILE.exists():
        return None
    with open(PIN_FILE, encoding="utf-8") as handle:
        pins = json.load(handle)
    source = pins.get("sources", {}).get(language)
    return source["url"] if source else None


def tool_versions():
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__ if onnxruntime else None,
        "converter": "scripts/fasttext_to_onnx.py",
    }


def write_metadata(metadata_path, language, onnx_name, checksum, source_url,
                   source_sha256, source_model, opset_version, command):
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata.update({
        "version": now,
        "url": source_url,
        "language": language,
        "type": "onnx",
        "file": onnx_name,
        "checksum": checksum,
        "cached_at": now,
        "source_model": source_model,
        "conversion_method": "fasttext_to_onnx.py",
        "opset_version": opset_version,
        "provenance": {
            "source_url": source_url,
            "source_sha256": source_sha256,
            "command": command,
            "tool_versions": tool_versions(),
        },
    })
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert one fastText .vec file to ONNX (single language)"
    )
    parser.add_argument("--language", required=True,
                        help="ISO language code, e.g. en (writes models/{lang})")
    parser.add_argument("--vec", required=True,
                        help="Path to the unpacked .vec file (no network access)")
    parser.add_argument("--vocab-size", type=int, default=100000,
                        help="Maximum vocabulary size (default: 100000, matches active models)")
    parser.add_argument("--repo-root", default=".",
                        help="Repository root used to resolve models/{lang} (default: .)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory override (default: {repo-root}/models/{lang})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing fasttext.{lang}.onnx/.vocab.json")
    args = parser.parse_args(argv)

    language = args.language
    if not re.fullmatch(r"[a-z]{2,3}", language):
        parser.error(f"--language must be a lowercase 2-3 letter code, got: {language!r}")

    vec_path = Path(args.vec)
    if not vec_path.is_file():
        print(f"Error: input .vec not found: {vec_path}", file=sys.stderr)
        return 1

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else repo_root / "models" / language
    onnx_path = out_dir / f"fasttext.{language}.onnx"
    vocab_path = out_dir / f"fasttext.{language}.vocab.json"
    metadata_path = out_dir / "metadata.json"

    if not args.force and (onnx_path.exists() or vocab_path.exists()):
        existing = ", ".join(str(p) for p in (onnx_path, vocab_path) if p.exists())
        print(f"Error: refusing to overwrite {existing}; pass --force to replace",
              file=sys.stderr)
        return 2

    word_to_idx, embeddings, vec_meta = fasttext_to_onnx.parse_fasttext_vec(
        str(vec_path), vocab_size=args.vocab_size
    )
    model = fasttext_to_onnx.create_onnx_model(embeddings, word_to_idx)

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(onnx_path))
    fasttext_to_onnx.save_vocabulary(word_to_idx, str(vocab_path))

    checksum = sha256_file(onnx_path)
    source_url = upstream_url_for(language) or vec_path.resolve().as_uri()
    source_sha256 = sha256_file(vec_path)
    opset_version = model.opset_import[0].version
    command = " ".join(sys.argv)

    write_metadata(metadata_path, language, onnx_path.name, checksum, source_url,
                   source_sha256, vec_path.name, opset_version, command)

    print(f"wrote {onnx_path} ({embeddings.shape[0]} x {embeddings.shape[1]})")
    print(f"wrote {vocab_path}")
    print(f"updated {metadata_path}")
    print(f"sha256: {checksum}")
    print("Reminder: manifest.json/registry.json were NOT updated; "
          "regenerate explicitly with scripts/generate_manifest.rb and "
          "scripts/build_registry.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
