#!/usr/bin/env python3
"""Extract unique (typo, correction) word pairs from the GitHub Typo
Corpus jsonl.gz, split into nonword/realword by 100k-vocab membership
(TODO.compare/1: the suggesters benchmark dataset).

    python scripts/extract_suggest_pairs.py --lang en [--max 5000]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--max", type=int, default=5000, help="pairs per class")
    args = ap.parse_args()

    vocab = json.loads((REPO / f"models/{args.lang}/fasttext.{args.lang}.vocab.json").read_text())
    vocab = vocab.get("word_to_idx", vocab)
    lang_code = {"en": "eng", "de": "deu", "es": "spa", "fr": "fra",
                 "pt": "por", "ru": "rus"}.get(args.lang, args.lang)

    # reuse the canonical extraction (fetch_corpus.extract_pairs collapses
    # each edit to (typo, correction, count) with the single-diff token
    # guard) - same rules as the Phase-0 evidence, both classes kept
    sys.path.insert(0, str(REPO / "scripts"))
    from fetch_corpus import extract_pairs, LANG_MAP  # noqa: E402
    lang_keys = [args.lang]  # extract_pairs keys by our code (LANG_MAP: corpus->ours)
    per_lang, _meta = extract_pairs(REPO / "eval/corpora/github-typo-corpus.v1.0.0.jsonl.gz")
    nonword, realword = Counter(), Counter()
    for key in lang_keys:
        for (s, t), w in per_lang.get(key, Counter()).items():
            s, t = s.lower(), t.lower()
            if s == t or not t.isalpha() or not s.isalpha():
                continue
            if s in vocab and t in vocab:
                realword[(s, t)] += w
            elif s not in vocab and t in vocab:
                nonword[(s, t)] += w

    def emit(pairs, klass):
        top = pairs.most_common(args.max)
        out = REPO / f"eval/realword/{args.lang}.suggest-{klass}.json"
        out.write_text(json.dumps(
            {"spec": "kotoshu.suggest-benchmark/v1", "language": args.lang,
             "klass": klass, "n": len(top),
             "pairs": [{"typo": s, "correction": t, "weight": w} for (s, t), w in top]},
            ensure_ascii=False, indent=1))
        print(f"{args.lang} {klass}: {len(pairs)} unique pairs -> top {len(top)} frozen to {out.name}")

    emit(nonword, "nonword")
    emit(realword, "realword")


if __name__ == "__main__":
    main()
