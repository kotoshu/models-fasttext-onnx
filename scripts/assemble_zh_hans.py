#!/usr/bin/env python3
"""Assemble the zh-Hans-CN quality corpus + measure the coverage gate.

Inputs: the jieba-segmented simp-only Wikipedia corpus (plan 20) plus
quality-filtered Common Crawl (jed351 CCF, C4 heuristics - the owner
law permits quality-filtered passes) segmented the same way. Output:
one merged training corpus + the plan-21 coverage verdict against the
crawl reference top-20k (gate: >= 99%; the plan-20 Wikipedia-only
model scored 85.7% and is thereby barred from production).

Usage:
  python scripts/assemble_zh_hans.py       --ccf eval/corpus/ccf-shard0.jsonl       --wiki eval/corpus/zh-hans-wiki.txt       --out eval/corpus/zh-hans-quality.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from segment_variants import classify  # noqa: E402

GATE = 0.99


def iter_jsonl(path: Path):
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if isinstance(text, str):
                yield text


def simp_reference(top_n: int = 20000) -> list[str]:
    """The OpenCC-verified Simplified reference: crawl top-N words that
    are (a) alpha, (b) contain at least one CJK char, (c) UNCHANGED
    under trad->simp conversion (a word like 位於 that converts is a
    Traditional word and belongs to the Hant gate, not this one). The
    old probe-char filter leaked Traditional words into the reference
    and understated coverage."""
    from opencc import OpenCC
    import json as _json
    t2s = OpenCC("t2s").convert
    vocab = _json.loads((REPO_ROOT / "models/zh/fasttext.zh.vocab.json").read_text())["word_to_idx"]
    cjk = lambda ch: "\u3400" <= ch <= "\u9fff"
    out = []
    for w in sorted(vocab, key=vocab.get):
        if len(out) >= top_n:
            break
        if not w.isalpha() or not any(cjk(c) for c in w):
            continue
        if t2s(w) == w:
            out.append(w)
    return out


def coverage(out_path: Path) -> float:
    """Streaming: unique corpus tokens only (never the whole corpus in RAM)."""
    ref = simp_reference()
    have = set()
    with out_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            have.update(line.split())
    missing = [w for w in ref if w not in have]
    return 1.0 - len(missing) / max(len(ref), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ccf", type=Path, required=True)
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=float, default=GATE)
    parser.add_argument("--ccf-max-mb", type=int, default=0, help="bound the CCF slice (0 = unbounded)")
    args = parser.parse_args()

    import jieba
    try:
        jieba.enable_parallel(8)
    except Exception as exc:
        print("jieba parallel unavailable:", exc)

    kept = 0
    with args.out.open("w", encoding="utf-8") as out:
        for line in args.wiki.read_text(encoding="utf-8", errors="replace").splitlines():
            if classify(line) == "simp":
                out.write(line + "\n")
                kept += 1
        for i, text in enumerate(iter_jsonl(args.ccf)):
            if args.ccf_max_mb and i * 40 > args.ccf_max_mb * 1024:
                print(f"ccf slice bound reached ({args.ccf_max_mb} MB of jsonl)")
                break
            for para in text.splitlines():
                words = [w for w in jieba.cut(para) if w.strip()]
                if not words:
                    continue
                line = " ".join(words)
                if classify(line) == "simp":
                    out.write(line + "\n")
                    kept += 1
    size_mb = args.out.stat().st_size / 2**20
    cov = coverage(args.out)
    print(f"kept lines: {kept:,} | corpus: {size_mb:.0f} MB | "
          f"coverage of crawl top-20k simp: {cov:.1%} (gate {args.gate:.0%})")
    if cov < args.gate:
        print(f"GATE FAIL: coverage {cov:.1%} < {args.gate:.0%} - add more shards", file=sys.stderr)
        return 2
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
