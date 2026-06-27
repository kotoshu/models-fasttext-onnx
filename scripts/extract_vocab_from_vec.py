#!/usr/bin/env python3
"""
Extract a .vocab.json sibling from a FastText .vec file.

The active 9 ONNX models in this repo were converted before
fasttext_to_onnx.py started emitting vocab.json siblings. Re-running
the full conversion is wasteful (each .onnx is ~114 MB and unchanged);
this script reads the .vec, takes the same first-N words in file order,
and emits a vocab.json in the exact format OnnxModel.from_file expects.

Usage:
    python3 extract_vocab_from_vec.py <vec_file> --vocab-size N --output PATH

Example:
    python3 extract_vocab_from_vec.py cc.en.300.vec \\
        --vocab-size 100000 \\
        --output models/en/fasttext.en.vocab.json

Correctness depends on the .vec word order matching what the original
fasttext_to_onnx.py conversion used (it read the .vec top-to-bottom and
stopped at --vocab-size). Spot-check by running OnnxModel.from_file and
confirming a known word like "the" resolves to a plausible embedding.
"""

import argparse
import json
import sys
from pathlib import Path


def parse_vec_vocab(vec_file_path, vocab_size=None):
    """Read a FastText .vec and return word -> row index for the first
    `vocab_size` words (or all words if vocab_size is None)."""
    word_to_idx = {}
    with open(vec_file_path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        total_words, dim = int(header[0]), int(header[1])
        cap = vocab_size if vocab_size is not None else total_words
        for i, line in enumerate(f):
            if i >= cap:
                break
            word = line.split(" ", 1)[0]
            word_to_idx[word] = i
    return word_to_idx, dim


def main():
    parser = argparse.ArgumentParser(
        description="Extract vocab.json from a FastText .vec file."
    )
    parser.add_argument("vec_file", help="Path to the FastText .vec file")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Maximum number of words to include (default: all)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write vocab.json",
    )
    args = parser.parse_args()

    vec_path = Path(args.vec_file)
    if not vec_path.exists():
        print(f"ERROR: {vec_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {vec_path}...")
    word_to_idx, dim = parse_vec_vocab(vec_path, args.vocab_size)
    print(f"  Loaded {len(word_to_idx)} words (dim={dim})")

    vocab_data = {
        "vocab_size": len(word_to_idx),
        "word_to_idx": word_to_idx,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False)
    print(f"Wrote {out_path} ({len(word_to_idx)} words)")


if __name__ == "__main__":
    main()
