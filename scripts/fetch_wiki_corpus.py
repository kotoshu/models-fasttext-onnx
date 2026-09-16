#!/usr/bin/env python3
"""Fetch the Phase-1 Wikipedia corpus shard for ctx-LM training (plan 16).

The corpus is the licensed source for the bigram trainer
(scripts/train_ctx_lm.py). Wikimedia dumps are CC-BY-SA-4.0; HuggingFace
mirrors the same content. We pull a single train shard of the
20231101.en config (~420 MB = a few million Wikipedia article
paragraphs ≈ hundreds of millions of tokens; more than enough for a
bigram LM).

Discovered at runtime via the HF datasets tree API so a future HF
re-organization only needs a re-run, not a code change. Writes a
LICENSE sidecar (eval/corpus/{lang}.LICENSE.json) the trainer
manifest cites — provenance for downstream licensing review.

The existing scripts/fetch_corpus.py is the GitHub Typo Corpus
fetcher (different purpose); this is the Wikipedia fetch for the
ctx-LM trainer specifically.

Usage:
  python scripts/fetch_wiki_corpus.py [--lang en] [--shards 1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "eval" / "corpus"
HF_API = "https://huggingface.co/api/datasets/{ds}/tree/main/{cfg}"
HF_RESOLVE = "https://huggingface.co/datasets/{ds}/resolve/main/{path}"


def list_shards(lang: str, config_date: str) -> list[dict]:
    url = HF_API.format(ds="wikimedia/wikipedia", cfg=f"{config_date}.{lang}")
    req = urllib.request.Request(url, headers={"User-Agent": "kotoshu-models/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return [
        {"path": f"{config_date}.{lang}/{x['path']}", "size": x.get("size", 0)}
        for x in data
        if x["type"] == "file" and x["path"].startswith("train-") and x["path"].endswith(".parquet")
    ]


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "kotoshu-models/1.0"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        with dest.open("wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", default="en")
    parser.add_argument("--config", default="20231101")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"listing {args.lang} shards for config {args.config}...")
    shards = list_shards(args.lang, args.config)
    if not shards:
        print("no shards found", file=sys.stderr)
        return 1
    shards.sort(key=lambda s: s["path"])
    selected = shards[: args.shards]
    for s in selected:
        print(f"  {s['path']}  {s['size'] / 2**20:.1f} MB")

    started = time.time()
    for s in selected:
        local = args.out / Path(s["path"]).name
        if local.exists() and local.stat().st_size == s["size"]:
            print(f"already have {local.name}, skipping")
            continue
        url = HF_RESOLVE.format(ds="wikimedia/wikipedia", path=s["path"])
        print(f"downloading {Path(s['path']).name}...")
        download(url, local)
    print(f"fetched in {time.time() - started:.1f}s -> {args.out}")

    (args.out / f"{args.lang}.LICENSE.json").write_text(json.dumps({
        "language": args.lang,
        "source": "HuggingFace mirror of wikimedia/wikipedia (20231101.en)",
        "license": "CC-BY-SA-4.0",
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shards": [s["path"] for s in selected],
        "note": "Wikimedia text content under CC-BY-SA-4.0; attribution and "
                "share-alike apply to derived artifacts. The trained ctx-LM "
                "counts are statistical, not textual, but the manifest cites "
                "this provenance for downstream licensing review.",
    }, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
