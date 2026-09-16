#!/usr/bin/env python3
"""Extract REAL-WORD error pairs (with sentence context) from the typo corpus.

`fetch_corpus.extract_pairs` collapses each corpus edit to a context-free
(typo, correction, count) triple — the right shape for retrieval training,
but real-word DETECTION evaluation needs the sentence around the error:
the detector scores the in-context word against its confusion set, and
the FP side needs clean sentences to measure wrong-flag rates.

Real-word class: BOTH sides of the edit are in the language's full-tier
vocab — the error a dictionary-gated checker can never see (en "I want
to each rice", zh homophone selections). Extraction rules are the
canonical ones (fetch_corpus.LANG_MAP + _word_pair, same single-diff
token guard); this script only ADDS what those rules drop: the source
sentence, the diff token index, and the vocab-membership split.

Outputs (per language):
  eval/realword/{lang}.json        pairs with contexts + counts
  eval/realword/{lang}.clean.jsonl corrected sentences (FP probe set)

Usage:
  python scripts/extract_realword_pairs.py --lang en \
      --corpus eval/corpora/github-typo-corpus.v1.0.0.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_corpus import LANG_MAP, _word_pair  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / "eval/corpora/github-typo-corpus.v1.0.0.jsonl.gz"
MAX_CONTEXTS = 3  # representative sentences per unique pair


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_vocab(lang: str) -> set[str]:
    path = REPO_ROOT / f"models/{lang}/fasttext.{lang}.vocab.json"
    data = json.loads(path.read_text())
    return set(data["word_to_idx"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True, help=f"one of {sorted(set(LANG_MAP.values()))}")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--max-contexts", type=int, default=MAX_CONTEXTS)
    args = parser.parse_args()

    if args.lang not in LANG_MAP.values():
        parser.error(f"--lang must be one of {sorted(set(LANG_MAP.values()))}")
    vocab = load_vocab(args.lang)
    corpus_langs = {k for k, v in LANG_MAP.items() if v == args.lang}

    stats = Counter()
    # (typo, correction) -> {"count": n, "contexts": [(sentence, idx), ...]}
    realword: dict[tuple[str, str], dict] = {}
    clean_sentences: set[str] = set()

    with gzip.open(args.corpus, "rt", encoding="utf-8") as fh:
        for line in fh:
            stats["commits"] += 1
            obj = json.loads(line)
            for edit in obj.get("edits", []):
                src = edit.get("src") or {}
                if src.get("lang", "") not in corpus_langs:
                    continue
                tgt = edit.get("tgt") or {}
                src_toks = src.get("text", "").split()
                tgt_toks = tgt.get("text", "").split()
                if not src_toks or len(src_toks) != len(tgt_toks):
                    continue
                diffs = [(i, a, b) for i, (a, b) in enumerate(zip(src_toks, tgt_toks)) if a != b]
                if len(diffs) != 1:
                    continue
                idx, a, b = diffs[0]
                pair = _word_pair(a, b)
                if pair is None:
                    continue
                stats["edits_used"] += 1
                typo, correction = pair
                if typo in vocab and correction in vocab:
                    stats["edits_realword"] += 1
                    entry = realword.setdefault((typo, correction), {"count": 0, "contexts": []})
                    entry["count"] += 1
                    if len(entry["contexts"]) < args.max_contexts:
                        entry["contexts"].append({"sentence": src.get("text", ""), "idx": idx})
                else:
                    stats["edits_nonword_side"] += 1
                clean_sentences.add(tgt.get("text", ""))

    out_dir = REPO_ROOT / "eval/realword"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_out = sorted(
        (
            {"typo": t, "correction": c, "count": e["count"], "contexts": e["contexts"]}
            for (t, c), e in realword.items()
        ),
        key=lambda p: (-p["count"], p["typo"], p["correction"]),
    )
    doc = {
        "language": args.lang,
        "corpus": {"name": "GitHub Typo Corpus", "version": "1.0.0", "path": str(args.corpus.relative_to(REPO_ROOT)), "sha256": sha256_of(args.corpus)},
        "vocab": {"tier": "full", "sha256": sha256_of(REPO_ROOT / f"models/{args.lang}/fasttext.{args.lang}.vocab.json")},
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": dict(stats),
        "n_pairs_unique": len(pairs_out),
        "pairs": pairs_out,
    }
    (out_dir / f"{args.lang}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    with (out_dir / f"{args.lang}.clean.jsonl").open("w") as fh:
        for sentence in sorted(clean_sentences):
            fh.write(sentence + "\n")

    print(
        f"{args.lang}: edits_used={stats['edits_used']} "
        f"realword_pairs={len(realword)} realword_edits={stats['edits_realword']} "
        f"clean_sentences={len(clean_sentences)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
