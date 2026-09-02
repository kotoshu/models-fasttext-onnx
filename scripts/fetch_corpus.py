#!/usr/bin/env python3
"""Fetch the GitHub Typo Corpus v1.0.0 and extract per-language word pairs.

Downloads the corpus (Hagiwara & Mita, LREC 2020; 350k+ real human typo
corrections, 15+ languages) into eval/corpora/ (gitignored), verifies its
sha256, and extracts (typo, correction) word pairs for the 9 languages the
tier eval covers. The corpus itself is never committed; only the derived
benchmark reports (eval/reports/corpus.{lang}.json) are.

Sources, tried in order:
1. The official S3 URL from the project README.
2. A Wayback Machine snapshot of that exact URL (the S3 bucket has been
   decommissioned; the snapshot is of the original file).

The release publishes no checksum of its own, so EXPECTED_SHA256 is
self-recorded from the archived copy of the official file and pinned here
to detect corruption/tampering on future re-downloads.

Extraction: one edit is a pair of before/after text snippets. We keep the
edit when src/tgt have the same whitespace token count and exactly one
differing token position. The differing token pair becomes a word pair if
both sides (after symmetric punctuation stripping and lowercasing) are
pure letters of length <= 16; for longer tokens (CJK snippets where the
whole sentence is one "token") the common prefix/suffix is stripped and
the remaining differing cores are used when both are <= 6 letters.
Pairs are deduplicated with occurrence counts.

Usage:
  python3 scripts/fetch_corpus.py --repo-root .
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

CORPUS_FILENAME = "github-typo-corpus.v1.0.0.jsonl.gz"
CORPUS_VERSION = "1.0.0"
CORPUS_URL = "https://github-typo-corpus.s3.amazonaws.com/data/" + CORPUS_FILENAME
WAYBACK_URL = (
    "https://web.archive.org/web/20260217103607/"
    "https://github-typo-corpus.s3.amazonaws.com/data/" + CORPUS_FILENAME
)
# Self-recorded (the release publishes no checksum); guards re-downloads.
EXPECTED_SHA256 = "98219d275a49ad28fca832891876c255581b890d893b6d9c974ceb454ba2909d"

# ISO 639-3 corpus codes -> our language codes. cmn-hant (traditional) is
# skipped: the zh fastText vocabulary is Simplified-dominant.
LANG_MAP = {
    "eng": "en",
    "deu": "de",
    "spa": "es",
    "fra": "fr",
    "por": "pt",
    "rus": "ru",
    "jpn": "ja",
    "kor": "ko",
    "cmn-hans": "zh",
}

_STRIP = "\"'`*_=~|.,;:!?()[]{}<>«»„“”‘’…·–—„\t"
MAX_WORD = 16  # token-level pair when both sides are words of this length
MAX_CJK_CORE = 6  # isolated diff core for sentence-like (CJK) tokens


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _download(sources: list[str], dest: Path) -> str:
    last_err: Exception | None = None
    for label, url in sources:
        try:
            print(f"downloading from {label}: {url}")
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs
            tmp.replace(dest)
            return label
        except Exception as exc:  # noqa: BLE001 - try next source
            print(f"  failed: {exc}", file=sys.stderr)
            last_err = exc
    raise RuntimeError(f"could not download corpus from any source: {last_err}")


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_token(tok: str) -> str:
    return tok.strip(_STRIP).lower()


def _word_pair(src_tok: str, tgt_tok: str) -> tuple[str, str] | None:
    """Return (typo, correction) for one differing token pair, or None."""
    a = _clean_token(src_tok)
    b = _clean_token(tgt_tok)
    if not a or not b or a == b:
        return None
    if a.isalpha() and b.isalpha() and len(a) <= MAX_WORD and len(b) <= MAX_WORD:
        return a, b
    # Sentence-like token (typical for zh/ja, no whitespace): isolate the
    # changed core by stripping the common prefix/suffix.
    if not (a.isalpha() and b.isalpha()):
        return None
    p = 0
    while p < len(a) and p < len(b) and a[p] == b[p]:
        p += 1
    s = 0
    while s < len(a) - p and s < len(b) - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    ta = a[p : len(a) - s]
    tb = b[p : len(b) - s]
    if (
        ta
        and tb
        and ta != tb
        and ta.isalpha()
        and tb.isalpha()
        and len(ta) <= MAX_CJK_CORE
        and len(tb) <= MAX_CJK_CORE
    ):
        return ta, tb
    return None


def extract_pairs(corpus_path: Path) -> tuple[dict[str, Counter], dict]:
    pairs: dict[str, Counter] = {lang: Counter() for lang in sorted(set(LANG_MAP.values()))}
    stats = {"commits": 0, "edits": 0, "edits_used": 0, "edits_lang_mismatch": 0}
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            stats["commits"] += 1
            obj = json.loads(line)
            for edit in obj.get("edits", []):
                stats["edits"] += 1
                src = edit.get("src") or {}
                lang = LANG_MAP.get(src.get("lang", ""))
                if lang is None:
                    stats["edits_lang_mismatch"] += 1
                    continue
                tgt = edit.get("tgt") or {}
                src_toks = src.get("text", "").split()
                tgt_toks = tgt.get("text", "").split()
                if not src_toks or len(src_toks) != len(tgt_toks):
                    continue
                diffs = [(a, b) for a, b in zip(src_toks, tgt_toks) if a != b]
                if len(diffs) != 1:
                    continue
                pair = _word_pair(*diffs[0])
                if pair is None:
                    continue
                stats["edits_used"] += 1
                pairs[lang][pair] += 1
    return pairs, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch GitHub Typo Corpus + extract per-language word pairs")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    parser.add_argument("--force", action="store_true", help="re-extract pairs even if already present")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    corpora_dir = repo / "eval" / "corpora"
    corpora_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpora_dir / CORPUS_FILENAME

    source_label = "cached"
    if not corpus_path.exists():
        source_label = _download([("official-s3", CORPUS_URL), ("wayback", WAYBACK_URL)], corpus_path)

    digest = _sha256(corpus_path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"{corpus_path}: sha256 {digest} != pinned {EXPECTED_SHA256} — refusing to use this file"
        )
    print(f"sha256 OK ({digest}); source={source_label}")

    print("extracting pairs (streams the whole corpus, takes a few minutes)...")
    pairs, stats = extract_pairs(corpus_path)

    corpus_meta = {
        "name": "GitHub Typo Corpus",
        "version": CORPUS_VERSION,
        "url": CORPUS_URL,
        "retrieved_from": WAYBACK_URL if source_label == "wayback" else CORPUS_URL,
        "sha256": digest,
        "license": "MIT (repository license); individual texts follow their source repositories' terms",
        "citation": "Hagiwara & Mita, 'GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors', LREC 2020 (arxiv 1911.12893)",
    }

    for lang in sorted(pairs):
        counter = pairs[lang]
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        out = {
            "language": lang,
            "corpus": corpus_meta,
            "extraction": {
                "rule": "same token count, exactly one differing token, both sides pure letters (<=16 chars, or <=6-char diff core for sentence-like CJK tokens), lowercased, symmetric punctuation stripped",
                "deduplicated": True,
            },
            "extraction_stats": stats,
            "n_pairs_unique": len(ordered),
            "pairs": [[t, c, n] for (t, c), n in ordered],
        }
        path = corpora_dir / f"{lang}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {lang}: {len(ordered)} unique pairs -> {path.relative_to(repo)}")

    meta = {"corpus": corpus_meta, "source_label": source_label, "stats": stats, "fetched_at": iso_now()}
    (corpora_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
