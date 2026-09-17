#!/usr/bin/env python3
"""Segment a zh Wikipedia shard and split it into script-variant corpora.

The owner directive (2026-09-16): zh splits into zh-Hans-CN and zh-Hant-TW/
zh-Hant-HK models because the regions differ by vocabulary, not just script,
and the shipped zh tiers are script-mixed (22.2% Traditional-only lines in
the source). This script produces the per-variant training text from the
mixed Wikipedia shard: jieba word segmentation (parallel), then line-level
script classification (simplified-only / traditional-only / mixed) using
disjoint probe sets. Mixed-script lines are DROPPED from both variants
(neither model trains on the other script or on inconsistent text).

Variant vocabulary filtering (TW vs HK within Traditional) happens
downstream, not here - this is the script split only.

Usage:
  python scripts/segment_variants.py --shard eval/corpus/zh-wiki-train-00000-of-00006.parquet \
      --out-hans eval/corpus/zh-hans-wiki.txt --out-hant eval/corpus/zh-hant-wiki.txt
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

TRAD_ONLY = set("們學國說話對時門問間長東車馬鳥語書頭點風飛龍業葉藥廠廣歸鐵電態應擊麗麼後讓認識覺觀歡樹機橋歷豐醫雙銀錯錢關閱顯餘")
SIMP_ONLY = set("们学国说话对时间问门东车马鸟语书头点风飞龙业叶药厂广归铁电态应击丽么后让认识觉观欢树机桥历归丰医双银错钱关阅显余")


def classify(line: str) -> str:
    has_t = any(c in TRAD_ONLY for c in line)
    has_s = any(c in SIMP_ONLY for c in line)
    if has_t and not has_s:
        return "trad"
    if has_s and not has_t:
        return "simp"
    if has_t and has_s:
        return "mixed"
    return "none"  # digits/latin/no probe chars


def segment(shard: Path, out_hans: Path, out_hant: Path) -> None:
    import jieba
    import pyarrow.parquet as pq

    try:
        jieba.enable_parallel(8)
        print("jieba parallel: on (8)")
    except Exception as exc:
        print("jieba parallel unavailable, single-threaded:", exc)

    table = pq.read_table(str(shard), columns=["text"])
    stats = Counter()
    t0 = time.time()
    with out_hans.open("w", encoding="utf-8") as fh_hans, out_hant.open("w", encoding="utf-8") as fh_hant:
        for i, text in enumerate(table.column("text").to_pylist()):
            for para in text.splitlines():
                if not para.strip():
                    continue
                words = [w for w in jieba.cut(para) if w.strip()]
                if not words:
                    continue
                line = " ".join(words)
                kind = classify(line)
                stats[kind] += 1
                if kind == "simp":
                    fh_hans.write(line + "\n")
                elif kind == "trad":
                    fh_hant.write(line + "\n")
            if (i + 1) % 20000 == 0:
                el = time.time() - t0
                print(f"  {i + 1:,}/{table.num_rows:,} articles ({(i + 1) / el:.0f}/s)", flush=True)
    total = sum(stats.values())
    print(f"lines: {dict(stats)} of {total:,}")
    print(f"hans: {out_hans} ({out_hans.stat().st_size / 2**20:.0f} MB)")
    print(f"hant: {out_hant} ({out_hant.stat().st_size / 2**20:.0f} MB)")
    print(f"elapsed: {(time.time() - t0) / 60:.0f} min")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--out-hans", type=Path, required=True)
    parser.add_argument("--out-hant", type=Path, required=True)
    args = parser.parse_args()
    segment(args.shard, args.out_hans, args.out_hant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
