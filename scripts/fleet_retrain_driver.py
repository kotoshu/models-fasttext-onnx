#!/usr/bin/env python3
"""Fleet retrain driver (plan 21 A4, v3 audit list): ca fa hu lv pl sv uk vi.

Per language (all Latin-script - no segmentation needed): download a
Wikipedia shard, extract clean text lines, run the wordfreq coverage
gate (v3 normalization: casefold + language norms + contraction
split-match), then hand the corpus to the Modal trainer. The driver is
idempotent per language and logs one verdict line each.

Usage:
  nohup python3 scripts/fleet_retrain_driver.py > /tmp/fleet-retrain.log 2>&1 &
"""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

LANGS = ["ca", "fa", "hu", "lv", "pl", "sv", "uk", "vi"]
GATE = 0.99
WIKI_BASE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main"


def norm(word: str) -> str:
    w = word.casefold()
    return unicodedata.normalize("NFC", w)


def download_shard(lang: str, dest: Path) -> bool:
    import urllib.request
    import json as _json
    api = f"https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/main/20231101.{lang}"
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "kotoshu-models/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            files = _json.load(resp)
    except Exception as exc:
        print(f"[{lang}] tree listing failed: {exc}")
        return False
    pdfs = [f["path"] for f in files if "/train-" in f["path"] and f["path"].endswith(".parquet")]
    if not pdfs:
        print(f"[{lang}] no train shards")
        return False
    url = f"{WIKI_BASE}/{pdfs[0]}"
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[{lang}] shard downloaded: {dest.stat().st_size / 2**20:.0f} MB")
        return True
    except Exception as exc:
        print(f"[{lang}] download failed: {exc}")
        return False


def extract_lines(shard: Path, out: Path, max_articles: int = 120_000) -> int:
    import pyarrow.parquet as pq
    kept = 0
    with out.open("w", encoding="utf-8") as fh:
        pf = pq.ParquetFile(str(shard))
        n = 0
        for batch in pf.iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column("text").to_pylist():
                n += 1
                if n > max_articles:
                    return kept
                for line in text.splitlines():
                    line = line.strip()
                    # quality heuristic: real prose lines, not nav/refs junk
                    if len(line) >= 20 and sum(c.isalpha() for c in line) / max(len(line), 1) > 0.7:
                        fh.write(line + "\n")
                        kept += 1
    return kept


def coverage(corpus: Path, lang: str) -> float:
    from wordfreq import top_n_list
    vocab_raw = json.loads((REPO / f"models/{lang}/fasttext.{lang}.vocab.json").read_text())["word_to_idx"]
    vocab = {norm(w) for w in vocab_raw}
    have = set()
    with corpus.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            have.update(norm(w) for w in line.split())
    ref = [norm(w) for w in top_n_list(lang, 20000)]
    def covered(w):
        if w in have:
            return True
        if "'" in w:
            parts = [p for p in w.split("'") if p]
            return all(p in have for p in parts)
        return False
    missing = [w for w in ref if not covered(w)]
    return 1 - len(missing) / len(ref)


def main() -> int:
    for lang in LANGS:
        corpus = REPO / f"eval/corpus/fleet-{lang}.txt"
        shard = REPO / f"eval/corpus/fleet-{lang}-shard.parquet"
        if not corpus.exists():
            if not download_shard(lang, shard):
                continue
            kept = extract_lines(shard, corpus)
            print(f"[{lang}] corpus: {kept:,} lines, {corpus.stat().st_size / 2**20:.0f} MB")
            shard.unlink(missing_ok=True)
        cov = coverage(corpus, lang)
        print(f"[{lang}] training-corpus coverage of wordfreq top-20k: {cov:.1%}")
        if cov < GATE:
            print(f"[{lang}] GATE FAIL ({cov:.1%}) - more sources needed, skipping train")
            continue
        print(f"[{lang}] GATE PASS - launching Modal training (detached)")
        subprocess.Popen(
            ["modal", "run", "scripts/modal_train_fasttext.py",
             "--corpus", str(corpus), "--name", f"fleet-{lang}",
             "--epoch", "5", "--bucket", "2000000"],
            cwd=REPO, stdout=open(f"/tmp/modal-fleet-{lang}.log", "w"),
            stderr=subprocess.STDOUT,
        )
    print("DRIVER COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
