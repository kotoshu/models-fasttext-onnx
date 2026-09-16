#!/usr/bin/env python3
"""Extract output-matrix rows for the tier vocab from a fastText .bin (plan 15).

The tier artifacts are INPUT vectors (what a word looks like); a
conditional score P(w | context) needs the OUTPUT matrix — context
input vectors dotted with the candidate's output row, the skipgram
scoring fastText itself uses. That matrix lives only in the training
binary ([nwords, dim] f32 for unquantized crawl models); this script
walks the .bin structure (reusing export_buckets' header machinery),
skips the quant byte + input matrix, and captures ONLY the rows whose
word is in the tier vocab, cached as a .npy aligned to the tier vocab
order (missing words = zero row + a recorded hit rate).

Cache: eval/cache/{lang}.output.npy + .meta.json — not committed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_buckets import parse_bin_header, read_exact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--bin", type=Path, required=True)
    args = parser.parse_args()

    vocab = json.loads((REPO_ROOT / f"models/{args.lang}/fasttext.{args.lang}.vocab.json").read_text())["word_to_idx"]

    open_handle = gzip.open if str(args.bin).endswith(".gz") else open
    with open_handle(args.bin, "rb") as fh:
        header, words = parse_bin_header(fh)
        dim = header.args["dim"]

        if read_exact(fh, 1)[0] != 0:
            raise SystemExit("quantized model — layout not handled")

        n_input_rows = header.nwords + header.bucket
        to_skip = n_input_rows * dim * 4
        while to_skip > 0:
            chunk = fh.read(min(1 << 24, to_skip))
            if not chunk:
                raise EOFError("EOF in input matrix")
            to_skip -= len(chunk)

        out = np.zeros((len(vocab), dim), dtype=np.float32)
        hit = 0
        for i in range(header.nwords):
            row = read_exact(fh, dim * 4)
            word = words[i][0]
            if word in vocab:
                out[vocab[word]] = np.frombuffer(row, dtype="<f4")
                hit += 1

    cache_dir = REPO_ROOT / "eval/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{args.lang}.output.npy", out)
    meta = {
        "language": args.lang,
        "bin": str(args.bin),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dim": dim,
        "vocab_size": len(vocab),
        "rows_captured": hit,
        "hit_rate": round(hit / len(vocab), 4),
    }
    (cache_dir / f"{args.lang}.output.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
