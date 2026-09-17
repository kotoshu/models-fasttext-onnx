#!/usr/bin/env python3
"""Assemble the zh-Hant-TW / zh-Hant-HK quality corpora + coverage gate.

Per plan 21 (corrected): the substrate is CC-100 zh-Hant (script-correct
crawl scale) + region-anchoring sources (TW: taiwan-corpus-zhtw;
HK: the LegCo Hansard) + OpenCC conversions of the QUALITY Simplified
corpus (s2twp converts vocabulary: software->software-TW; s2hk for HK).
The probe-filtered mixed-wiki extract is REJECTED (regionally
uncontrolled).

The coverage reference is the Hant form of the independent word list:
wordfreq zh top-20k converted with the same regional profile - a TW
model is gated on TW-convention words, an HK model on HK-convention
words.

Usage:
  python scripts/assemble_zh_hant.py --variant zh-Hant-TW \
      --cc100 eval/corpus/cc100-zh-hant/train-00000.parquet \
      --hans eval/corpus/zh-hans-quality.txt \
      --tw-net eval/corpus/taiwan-net-zhtw.jsonl \
      --hansard ~/src/kotoshu/hk-hansard/data/text \
      --out eval/corpus/zh-hant-tw.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from segment_variants import classify  # noqa: E402

PROFILES = {
    "zh-Hant-TW": "s2twp",
    "zh-Hant-HK": "s2hk",
}
GATE = 0.99


def iter_parquet_lines(path: Path, cap_rows: int):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    n = 0
    for batch in pf.iter_batches(batch_size=65536, columns=["line"]):
        for row in batch.column("line").to_pylist():
            n += 1
            if n > cap_rows:
                return
            yield row


def iter_jsonl_texts(paths):
    import json
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if isinstance(text, str):
                yield text


def iter_hansard_files(directory: Path, cap_files: int):
    files = sorted(directory.glob("*.txt"))[:cap_files]
    for f in files:
        yield f.read_text(encoding="utf-8", errors="replace")


def coverage(out_path: Path, profile: str) -> float:
    from opencc import OpenCC
    from wordfreq import top_n_list
    conv = OpenCC(profile).convert
    ref = []
    for w in top_n_list("zh", 20000):
        hw = conv(w)
        if hw and hw != w:  # only words that HAVE a Hant form
            ref.append(hw)
    have = set()
    with out_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            have.update(line.split())
    missing = [w for w in ref if w not in have]
    return 1.0 - len(missing) / max(len(ref), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(PROFILES), required=True)
    parser.add_argument("--cc100", type=Path, required=True)
    parser.add_argument("--hans", type=Path, required=True)
    parser.add_argument("--tw-net", type=Path, default=None)
    parser.add_argument("--hansard", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cc100-rows", type=int, default=400_000)
    parser.add_argument("--hansard-files", type=int, default=400)
    parser.add_argument("--gate", type=float, default=GATE)
    args = parser.parse_args()

    profile = PROFILES[args.variant]
    from opencc import OpenCC
    conv = OpenCC(profile).convert

    import jieba
    try:
        jieba.enable_parallel(8)
        print("jieba parallel: on")
    except Exception as exc:
        print("jieba parallel unavailable:", exc)

    kept = 0

    def emit(line: str):
        nonlocal kept
        line = line.strip()
        if not line:
            return
        if classify(line) in ("trad", "none"):
            out.write(line + "\n")
            kept += 1

    with args.out.open("w", encoding="utf-8") as out:
        # 1. CC-100 zh-Hant (script-correct crawl) - segment + keep trad lines
        for row in iter_parquet_lines(args.cc100, args.cc100_rows):
            for para in row.splitlines():
                words = [w for w in jieba.cut(para) if w.strip()]
                if words:
                    emit(" ".join(words))
        # 2. region anchor: TW-curated web text / HK Hansard
        if args.variant == "zh-Hant-TW" and args.tw_net:
            for text in iter_jsonl_texts([args.tw_net]):
                for para in text.splitlines():
                    words = [w for w in jieba.cut(para) if w.strip()]
                    if words:
                        emit(" ".join(words))
        if args.variant == "zh-Hant-HK" and args.hansard:
            for text in iter_hansard_files(args.hansard, args.hansard_files):
                for para in text.splitlines():
                    words = [w for w in jieba.cut(para) if w.strip()]
                    if words:
                        emit(" ".join(words))
        # 3. the QUALITY Hans corpus converted regionally (vocab included)
        for line in args.hans.read_text(encoding="utf-8", errors="replace").splitlines():
            emit(conv(line))

    size_mb = args.out.stat().st_size / 2**20
    cov = coverage(args.out, profile)
    print(f"kept lines: {kept:,} | corpus: {size_mb:.0f} MB | "
          f"coverage of {profile} reference: {cov:.1%} (gate {args.gate:.0%})")
    if cov < args.gate:
        print(f"GATE FAIL: {cov:.1%} < {args.gate:.0%}", file=sys.stderr)
        return 2
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
