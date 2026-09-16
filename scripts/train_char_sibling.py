#!/usr/bin/env python3
"""Train a short-ngram sibling fastText model + run the plan-18 feasibility probe.

The 8 bucket-rejected languages (ar cs fa he ja pl vi zh) fail because
the Common Crawl binaries are trained with minn=maxn=5 — short tokens
(CJK words especially) produce no stored subword rows. This harness
trains a sibling model with minn=1/maxn=3 over word-segmented text and
runs the existing bucket exporter gates against it, recording the
verdict honestly: retraining SOLVES row starvation, but a sibling
model's bucket rows live in a DIFFERENT vector space than the crawl
tiers the composition ranks against — cross-space cosine is
meaningless. A Procrustes alignment probe (shared-vocab anchors) is
included to quantify how much signal survives the space transfer.

Verdict recorded in eval/reports/{lang}.buckets.char-probe.json +
TODO.impl/18-cjk-char-buckets.md. Corpus inputs stay local
(.gitignore); models land in eval/cache/.

Usage (the zh probe as executed):
  # 1. segment a shard into word-per-line training text
  python scripts/train_char_sibling.py --lang zh segment \
      --parquet eval/corpus/zh-wiki-train-00000-of-00006.parquet \
      --out eval/corpus/zh-word-train-probe.txt --max-paragraphs 60000
  # 2. train the sibling (fastText skipgram, short ngrams)
  python scripts/train_char_sibling.py --lang zh train \
      --corpus eval/corpus/zh-word-train-probe.txt \
      --bin eval/cache/zh-char-probe.bin
  # 3. run the existing gates against the sibling
  python scripts/export_buckets.py --lang zh --bin eval/cache/zh-char-probe.bin \
      --k 32768,65536,131072
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# CJK Unified Ideographs (basic): the keep-filter for segmented tokens
def _cjk_or_alnum(w: str) -> str:
    return "".join(c for c in w if ("\u3400" <= c <= "\u9fff") or c.isalnum())


def segment(parquet: Path, out: Path, max_paragraphs: int) -> None:
    import jieba
    import pyarrow.parquet as pq
    table = pq.read_table(parquet, columns=["text"])
    rows = 0
    with out.open("w") as fh:
        for text in table.column("text").to_pylist():
            for para in text.splitlines():
                words = [w for w in (_cjk_or_alnum(x) for x in jieba.cut(para)) if w]
                if words:
                    fh.write(" ".join(words) + "\n")
                    rows += 1
            if rows >= max_paragraphs:
                break
    print(f"rows {rows} bytes {out.stat().st_size}")


def train(corpus: Path, bin_path: Path, dim: int, minn: int, maxn: int,
           bucket: int, epoch: int) -> None:
    import fasttext
    model = fasttext.train_unsupervised(
        str(corpus), model="skipgram", dim=dim, minn=minn, maxn=maxn,
        bucket=bucket, minCount=5, epoch=epoch, thread=8, ws=5, lr=0.05,
    )
    model.save_model(str(bin_path))
    print(f"trained {bin_path} words={len(model.get_words())} dim={model.get_dimension()}")


def align_probe(bin_path: Path, lang: str, top_pairs: int) -> None:
    """Procrustes-align the sibling space onto the mini tier space and
    rescore the demand pairs — quantifies how much signal survives the
    cross-space transfer (the structural finding of plan 18)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import numpy as np
    import fasttext
    from export_buckets import (parse_bin_header, read_input_matrix,
                                marked_ngram_occurrences, fasttext_hash,
                                load_tier, normalize)
    tier_vocab, tier_rows = load_tier(
        REPO_ROOT / f"models/{lang}/fasttext.{lang}.mini.onnx",
        REPO_ROOT / f"models/{lang}/fasttext.{lang}.mini.vocab.json")
    ft = fasttext.load_model(str(bin_path))
    anchors = [w for w in ft.get_words() if w in tier_vocab]
    A = np.stack([ft.get_word_vector(w) for w in anchors]).astype(np.float64)
    B = np.stack([tier_rows[tier_vocab[w]] for w in anchors]).astype(np.float64)
    W, *_ = np.linalg.lstsq(A, B, rcond=None)
    residual = float(np.linalg.norm(A @ W - B) / np.linalg.norm(B))

    with open(bin_path, "rb") as h:
        header, _words = parse_bin_header(h)
        rows, dim = read_input_matrix(h, header)
    rows = (rows.astype(np.float64) @ W).astype(np.float32)

    pairs = json.loads((REPO_ROOT / f"eval/corpora/{lang}.json").read_text())["pairs"][:top_pairs]
    unit = tier_rows / np.maximum(np.linalg.norm(tier_rows, axis=1, keepdims=True), 1e-30)
    word_by_row = {i: w for w, i in tier_vocab.items()}
    hits1 = hits5 = used = 0
    for typo, correct, _count in pairs:
        if correct not in tier_vocab:
            continue
        total = np.zeros(dim, dtype=np.float64)
        resolved = 0
        for ng in marked_ngram_occurrences(typo, header.args["minn"], header.args["maxn"]):
            total += rows[fasttext_hash(ng) % header.bucket]
            resolved += 1
        if resolved == 0:
            continue
        vec = normalize(total.astype(np.float32))
        if vec is None:
            continue
        sims = unit @ vec
        order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), word_by_row[i]))[:20]
        used += 1
        if word_by_row[order[0]] == correct:
            hits1 += 1
        if correct in [word_by_row[i] for i in order[:5]]:
            hits5 += 1
    print(json.dumps({
        "anchors": len(anchors),
        "procrustes_relative_residual": round(residual, 4),
        "aligned_probes_used": used,
        "aligned_intended_top1": round(hits1 / max(used, 1), 4),
        "aligned_intended_top5": round(hits5 / max(used, 1), 4),
    }))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    seg = sub.add_parser("segment")
    seg.add_argument("--parquet", type=Path, required=True)
    seg.add_argument("--out", type=Path, required=True)
    seg.add_argument("--max-paragraphs", type=int, default=60000)
    tr = sub.add_parser("train")
    tr.add_argument("--corpus", type=Path, required=True)
    tr.add_argument("--bin", type=Path, required=True)
    tr.add_argument("--dim", type=int, default=300)
    tr.add_argument("--minn", type=int, default=1)
    tr.add_argument("--maxn", type=int, default=3)
    tr.add_argument("--bucket", type=int, default=200000)
    tr.add_argument("--epoch", type=int, default=5)
    al = sub.add_parser("align-probe")
    al.add_argument("--bin", type=Path, required=True)
    al.add_argument("--lang", required=True)
    al.add_argument("--top-pairs", type=int, default=2000)
    args = parser.parse_args()
    if args.cmd == "segment":
        segment(args.parquet, args.out, args.max_paragraphs)
    elif args.cmd == "train":
        train(args.corpus, args.bin, args.dim, args.minn, args.maxn, args.bucket, args.epoch)
    else:
        align_probe(args.bin, args.lang, args.top_pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
