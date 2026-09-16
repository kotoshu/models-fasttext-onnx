#!/usr/bin/env python3
"""Build the per-language confusion-pair table (real-word detection, plan 15).

A confusion set is the runtime's answer to "which OTHER in-vocab words
could this valid word be a mis-selection of?" — the candidate pool the
context scorer ranks the observed word against (an/a, then/than,
seperate/separate; zh homophone pairs in the CJK phase).

Membership v1 = Damerau-Levenshtein distance <= 1 (substitution,
insertion, deletion, adjacent transposition). Generated via the
deletion-neighborhood index: every DL<=1 pair shares at least one
1-deletion variant, so bucketing words by their deletion variants and
verifying pairs inside buckets enumerates exactly the DL<=1 set without
an O(n^2) sweep. Keyboard-adjacency and phonetic-class sources are
additive follow-ons (the format stores per-pair sources so enrichment
never changes the runtime shape); corpus-derived pairs are deliberately
EXCLUDED — they are the evaluation set, and a table that memorized its
own benchmark reports a fake ceiling.

Output: eval/confusion/{lang}.json — symmetric table {word: [neighbors]}
so runtime lookup is one dict hit, plus sources {word: {neighbor:
["dl1"]}} for provenance.

Usage:
  python scripts/build_confusion_tables.py --lang en
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_vocab(lang: str) -> list[str]:
    path = REPO_ROOT / f"models/{lang}/fasttext.{lang}.vocab.json"
    return list(json.loads(path.read_text())["word_to_idx"])


def dl_distance_le1(a: str, b: str) -> bool:
    """Damerau-Levenshtein distance <= 1, without building a matrix."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(diff) == 1:
            return True
        # adjacent transposition: exactly two positions differ, swapped
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b, la, lb = b, a, lb, la  # a is the shorter word
    # deletion from b == insertion into a: exactly one char extra
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1 :]


def deletion_variants(word: str) -> list[str]:
    return [word[:i] + word[i + 1 :] for i in range(len(word))] if len(word) >= 2 else [word]


def build_confusions(vocab: list[str]) -> dict[str, dict[str, list[str]]]:
    index: dict[str, list[str]] = defaultdict(list)
    for word in vocab:
        for variant in deletion_variants(word):
            index[variant].append(word)
    table: dict[str, dict[str, list[str]]] = defaultdict(dict)
    seen: set[tuple[str, str]] = set()
    for words in index.values():
        if len(words) < 2:
            continue
        for i, w1 in enumerate(words):
            for w2 in words[i + 1 :]:
                if w1 == w2 or (w1, w2) in seen:
                    continue
                if dl_distance_le1(w1, w2):
                    seen.add((w1, w2))
                    table[w1][w2] = ["dl1"]
                    table[w2][w1] = ["dl1"]
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()

    vocab_path = REPO_ROOT / f"models/{args.lang}/fasttext.{args.lang}.vocab.json"
    vocab = load_vocab(args.lang)
    table = build_confusions(vocab)

    out_dir = REPO_ROOT / "eval/confusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "language": args.lang,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vocab": {"tier": "full", "sha256": sha256_of(vocab_path), "size": len(vocab)},
        "sources": {"dl1": "Damerau-Levenshtein <= 1 (substitution/insertion/deletion/transposition)"},
        "n_words_with_confusions": len(table),
        "n_pairs": sum(len(n) for n in table.values()) // 2,
        "table": table,
    }
    (out_dir / f"{args.lang}.json").write_text(json.dumps(doc, ensure_ascii=False) + "\n")
    print(
        f"{args.lang}: vocab={len(vocab)} words_with_confusions={len(table)} "
        f"pairs={doc['n_pairs']} mean_degree={doc['n_pairs'] * 2 / max(len(table), 1):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
