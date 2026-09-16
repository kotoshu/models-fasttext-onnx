#!/usr/bin/env python3
"""Train a per-language bigram context LM (plan 16, Phase 1 ctx-LM).

Consumes a corpus of plain paragraphs (.parquet / .jsonl[.gz] / .txt),
counts unigrams + bigrams restricted to words present in the tier vocab
(the scorer never asks about an out-of-vocab target), and emits the
artifact the Phase-1 gate scores against: unigram counts indexed by
vocab id, bigram counts indexed by a 32-bit hash of the (a, b) id
pair, plus a JSON manifest recording source + license + backoff alpha
+ tokenizer.

The artifact format is deliberately NOT yet the design doc's ONNX
shape: the gate decides whether the artifact matters - on PASS, the
unblock PRs repackage into the int8 ONNX tensors the doc specified
(fasttext.{lang}.ctx.onnx; q_scores, row_scale, ngram_ids) using the
npz arrays as the source of truth. Ship the source of truth first.

Why a bigram context window, not a full LM: the Phase-0 probe scored
each candidate as sum over the +-5 window of a context-fit signal -
the same shape, with the score now coming from a real P(w|neighbor)
estimate. Trigram escalation stays in the ladder per the design doc
if the gate fails.

Usage:
  python scripts/train_ctx_lm.py --lang en \
      --corpus eval/corpus/en-wiki-train-00000-of-00041.parquet \
      --out models/en/fasttext.en.ctx.npz \
      --source 'wikimedia/wikipedia 20231101.en shard 0' \
      --license CC-BY-SA-4.0
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

WORD_RE = re.compile(r"^[a-z]+(?:\'[a-z]+)*$")
BACKOFF_ALPHA = 0.4


def hash_pair(a: int, b: int) -> int:
    """32-bit hash of an ordered id pair; collision-safe at the size
    we will see in Phase 1 (~5M bigram types -> ~0.06% collision rate)."""
    h = hashlib.blake2b(f"{a}|{b}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "big")


def load_vocab_ids(lang: str) -> tuple[dict[str, int], list[str]]:
    path = REPO_ROOT / f"models/{lang}/fasttext.{lang}.vocab.json"
    data = json.loads(path.read_text())["word_to_idx"]
    ids = {w: i for i, (w, _) in enumerate(sorted(data.items(), key=lambda kv: kv[1])) if w.isalpha()}
    return ids, sorted(ids, key=ids.get)


def iter_corpus_texts(corpus_path: Path):
    if corpus_path.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(corpus_path, columns=["text"])
        for v in table.column("text").to_pylist():
            if isinstance(v, str):
                yield v
        return
    opener = gzip.open if str(corpus_path).endswith(".gz") else open
    with opener(corpus_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            for key in ("text", "body", "content", "article"):
                if isinstance(obj.get(key), str):
                    yield obj[key]
                    break


def tokenize(paragraph: str) -> list[str]:
    out: list[str] = []
    for tok in paragraph.lower().split():
        cleaned = "".join(c for c in tok if c.isalpha() or c == "'")
        if WORD_RE.match(cleaned):
            out.append(cleaned)
    return out


def train(unigrams, bigrams, vocab_ids, corpus_path):
    n_tokens = 0
    for text in iter_corpus_texts(corpus_path):
        ids = [vocab_ids[t] for t in tokenize(text) if t in vocab_ids]
        n_tokens += len(ids)
        unigrams.update(ids)
        bigrams.update(zip(ids, ids[1:]))
    return n_tokens, len(bigrams)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=BACKOFF_ALPHA)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", default="CC-BY-SA-4.0")
    args = parser.parse_args()

    vocab_ids, _ = load_vocab_ids(args.lang)
    out_path = args.out or (REPO_ROOT / f"models/{args.lang}/fasttext.{args.lang}.ctx.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    unigrams = Counter()
    bigrams = Counter()
    n_tokens, n_bigram_types = train(unigrams, bigrams, vocab_ids, args.corpus)

    vocab_size = max(vocab_ids.values()) + 1
    uni_arr = np.zeros(vocab_size, dtype=np.uint32)
    for wid, c in unigrams.items():
        uni_arr[wid] = c

    bg_keys = np.fromiter((hash_pair(a, b) for (a, b) in bigrams), dtype=np.uint32, count=len(bigrams))
    bg_counts = np.fromiter(bigrams.values(), dtype=np.uint32, count=len(bigrams))
    order = np.argsort(bg_keys)
    bg_keys = bg_keys[order]
    bg_counts = bg_counts[order]

    np.savez(out_path, unigram_counts=uni_arr, bigram_keys=bg_keys, bigram_counts=bg_counts)

    manifest = {
        "language": args.lang,
        "source": args.source,
        "license": args.license,
        "corpus_path": str(args.corpus),  # may be absolute or relative-to-cwd
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vocab_size": int(vocab_size),
        "tokens": int(n_tokens),
        "bigram_types": int(n_bigram_types),
        "backoff_alpha": args.alpha,
        "quantization": "fp32 (gate run); int8-per-row ONNX repackaging on unblock",
        "artifact_format": "npz: {unigram_counts uint32[vocab_size], bigram_keys uint32[n_types], bigram_counts uint32[n_types]}",
    }
    (out_path.parent / f"{out_path.stem}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"{args.lang}: tokens={n_tokens} unigram_types={len(unigrams)} "
          f"bigram_types={n_bigram_types} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
