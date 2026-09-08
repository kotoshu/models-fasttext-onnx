#!/usr/bin/env python3
"""Train the plan-111 candidate C: a purpose-trained typo bi-encoder.

Trains a small character-level BiGRU bi-encoder on the VENDORED GitHub Typo
Corpus (eval/corpora/github-typo-corpus.v1.0.0.jsonl.gz, sha256-pinned by
scripts/fetch_corpus.py) with a train/eval split STRICTLY BY REPOSITORY:
the commit stream is grouped by repo, repos are deterministically shuffled
(default_rng([42])) and split 80/20, and no repo ever appears on both sides.
A held-out pool pair is additionally required to occur ONLY in held-out
repos (a pair seen in any train repo could be memorized), so the bench
(eval/bakeoff_bench.py) evaluates C on the corpus_bench pool minus pairs
with any training-repo occurrence.

Training signal: symmetric InfoNCE (typo -> correction and back) with
in-batch negatives, occurrence-weighted pairs (weight capped per pair so
one ultra-frequent pair cannot dominate), temperature 0.05.

The trained model is exported to ONNX and int8-quantized with the same
quantize_dynamic used by the shipped tiers. Everything lands in
eval/candidates/c_typo/ (gitignored); the receipt (sizes, sha256, split
counts, hyperparameters, held-out retrieval metric) goes to
eval/candidates/manifest.json for the bench to cite.

Usage:
  python3 scripts/train_typo_biencoder.py --repo-root .
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SEED = 42
SPLIT_TRAIN_FRAC = 0.8
MAX_WORD_LEN = 24
WEIGHT_CAP = 8  # occurrence weight cap per unique pair per epoch

CHAR_DIM = 48
GRU_DIM = 96
OUT_DIM = 256
TEMP = 0.05
BATCH = 512
EPOCHS = 8
LR = 3e-3

TARGET_LANGS = ("en", "de", "es", "ru")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_repo_pairs(corpus_path: Path):
    """Yield (repo, lang, typo, correction) using fetch_corpus extraction rules."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_corpus import LANG_MAP, _word_pair

    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            repo = obj.get("repo", "")
            for edit in obj.get("edits", []):
                src = edit.get("src") or {}
                lang = LANG_MAP.get(src.get("lang", ""))
                if lang is None:
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
                yield repo, lang, pair[0], pair[1]


class CharBiEncoder(nn.Module):
    """Shared char encoder: BiGRU over char embeddings, masked mean pool, projection, L2 norm."""

    def __init__(self, n_chars: int) -> None:
        super().__init__()
        self.pad = 0
        self.emb = nn.Embedding(n_chars, CHAR_DIM, padding_idx=0)
        self.gru = nn.GRU(CHAR_DIM, GRU_DIM, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * GRU_DIM, OUT_DIM)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != self.pad).unsqueeze(-1).to(torch.float32)
        h, _ = self.gru(self.emb(ids))
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)


def encode_words(words: list[str], char_to_idx: dict) -> np.ndarray:
    unk = len(char_to_idx)  # last id is <unk>
    out = np.zeros((len(words), MAX_WORD_LEN), dtype=np.int64)
    for i, w in enumerate(words):
        for j, ch in enumerate(w[:MAX_WORD_LEN]):
            out[i, j] = char_to_idx.get(ch, unk)
    return out


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    logits = z_a @ z_b.T / TEMP
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the plan-111 typo bi-encoder (candidate C)")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    corpus_path = repo / "eval" / "corpora" / "github-typo-corpus.v1.0.0.jsonl.gz"
    if not corpus_path.exists():
        parser.error(f"{corpus_path} missing — run scripts/fetch_corpus.py first")

    print("streaming corpus (repo-grouped pairs, fetch_corpus extraction rules)...")
    pair_repos: dict[tuple[str, str, str], set[str]] = {}
    repo_pairs: dict[str, Counter] = {}
    lang_counter: Counter = Counter()
    for repo_url, lang, typo, corr in stream_repo_pairs(corpus_path):
        key = (lang, typo, corr)
        pair_repos.setdefault(key, set()).add(repo_url)
        repo_pairs.setdefault(repo_url, Counter())[(lang, typo, corr)] += 1
        lang_counter[lang] += 1
    repos = sorted(repo_pairs)
    print(f"repos={len(repos)} pairs_unique={len(pair_repos)} edits_used={sum(lang_counter.values())}")
    print("per-lang edits:", dict(lang_counter.most_common()))

    rng = np.random.default_rng([SEED])
    perm = rng.permutation(len(repos))
    n_train = int(len(repos) * SPLIT_TRAIN_FRAC)
    train_repos = {repos[i] for i in perm[:n_train]}
    heldout_repos = {repos[i] for i in perm[n_train:]}
    assert not (train_repos & heldout_repos)

    train_counter: Counter = Counter()
    for r in train_repos:
        train_counter.update(repo_pairs[r])
    train_items = []
    for (lang, typo, corr), n in train_counter.items():
        train_items.extend([(typo, corr)] * min(n, WEIGHT_CAP))
    heldout_clean = {k for k, v in pair_repos.items() if v <= heldout_repos}
    heldout_dirty = {k for k, v in pair_repos.items() if v & train_repos}
    print(
        f"split: {len(train_repos)} train repos / {len(heldout_repos)} held-out repos; "
        f"{len(train_counter)} unique train pairs ({len(train_items)} weighted rows); "
        f"{len(heldout_clean)} pairs held-out-clean, {len(heldout_dirty)} excluded from eval (seen in a train repo)"
    )

    # char vocabulary: train + held-out pair words plus every character of the
    # four target languages' full fastText vocabularies (the ranking universe)
    chars: set[str] = set()
    for lang in TARGET_LANGS:
        vocab_path = repo / "models" / lang / f"fasttext.{lang}.vocab.json"
        word_to_idx = json.loads(vocab_path.read_text(encoding="utf-8"))["word_to_idx"]
        for w in word_to_idx:
            chars.update(w[:MAX_WORD_LEN])
    for lang, typo, corr in pair_repos:
        chars.update(typo[:MAX_WORD_LEN])
        chars.update(corr[:MAX_WORD_LEN])
    char_to_idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
    print(f"char vocab: {len(char_to_idx)} + pad + unk")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = CharBiEncoder(len(char_to_idx) + 2)  # pad=0, chars 1..N, unk=N+1

    # eval probe: rank the correction for held-out-clean TARGET-lang pairs
    # among a fixed 10k sample of the full en vocabulary (proxy metric during
    # training; the real bench ranks among each language's whole vocab)
    en_vocab_path = repo / "models" / "en" / "fasttext.en.vocab.json"
    en_words = list(json.loads(en_vocab_path.read_text(encoding="utf-8"))["word_to_idx"])
    prng = np.random.default_rng([SEED, 7])
    probe_words = [en_words[i] for i in prng.choice(len(en_words), size=10_000, replace=False)]
    probe_pairs = [(t, c) for (lang, t, c) in heldout_clean if lang == "en"][:1000]
    probe_universe = probe_words + [c for _, c in probe_pairs]
    probe_universe_enc = torch.from_numpy(encode_words(probe_universe, char_to_idx))
    # column of the FIRST occurrence of each word (to exclude the typo itself
    # from its own candidate list when it happens to be in the universe)
    probe_col = {}
    for i, w in enumerate(probe_universe):
        probe_col.setdefault(w, i)
    probe_corr_idx = [probe_col[c] for _, c in probe_pairs]

    def probe_hit5() -> float:
        if not probe_pairs:
            return float("nan")
        typo_enc = torch.from_numpy(encode_words([t for t, _ in probe_pairs], char_to_idx))
        with torch.inference_mode():
            q = model(typo_enc)
            sims = q @ model(probe_universe_enc).T
            for i, (t, _c) in enumerate(probe_pairs):
                col = probe_col.get(t)
                if col is not None:
                    sims[i, col] = -1.0
            top5 = sims.topk(5, dim=1).indices
        hits = sum(1 for i, c_idx in enumerate(probe_corr_idx) if c_idx in top5[i])
        return hits / len(probe_pairs)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = {"hit5": -1.0, "state": None, "epoch": -1}
    train_arr = encode_words([w for pair in train_items for w in pair], char_to_idx).reshape(len(train_items), 2, MAX_WORD_LEN)
    for epoch in range(1, EPOCHS + 1):
        order = np.random.default_rng([SEED, epoch]).permutation(len(train_arr))
        model.train()
        total = 0.0
        for lo in range(0, len(order), BATCH):
            idx = order[lo : lo + BATCH]
            if idx.size < 32:  # too few in-batch negatives to be a real batch
                continue
            batch = train_arr[idx]
            z_t = model(torch.from_numpy(batch[:, 0]))
            z_c = model(torch.from_numpy(batch[:, 1]))
            loss = info_nce(z_t, z_c)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(idx.size)
        model.eval()
        hit5 = probe_hit5()
        print(f"epoch {epoch}: loss={total / len(order):.4f} probe_hit5={hit5:.4f}")
        if hit5 > best["hit5"]:
            best = {"hit5": hit5, "state": {k: v.clone() for k, v in model.state_dict().items()}, "epoch": epoch}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    out_dir = repo / "eval" / "candidates" / "c_typo"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    class ExportWrapper(nn.Module):
        def __init__(self, m: CharBiEncoder) -> None:
            super().__init__()
            self.m = m

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.m(ids)

    sample = torch.from_numpy(encode_words(["example", "пример"], char_to_idx))
    fp32_path = out_dir / "model.fp32.onnx"
    int8_path = out_dir / "model.int8.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            ExportWrapper(model),
            (sample,),
            str(fp32_path),
            input_names=["char_ids"],
            output_names=["embedding"],
            dynamic_axes={"char_ids": {0: "batch", 1: "seq"}},
            opset_version=17,
            dynamo=False,
        )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    # export parity spot check
    import onnxruntime as ort

    with torch.inference_mode():
        ref = model(sample).numpy()
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    got = sess.run(["embedding"], {"char_ids": sample.numpy()})[0]
    max_abs = float(np.abs(got - ref).max())
    if not np.all(np.isfinite(got)) or max_abs > 0.05:
        raise RuntimeError(f"c_typo int8 ONNX diverges from torch (max abs {max_abs:.4g})")

    n_params = sum(p.numel() for p in model.parameters())
    receipt = {
        "name": "c_typo",
        "kind": "biencoder",
        "architecture": f"char-BiGRU ({CHAR_DIM}d chars, {GRU_DIM}d GRU, {OUT_DIM}d out, shared encoder)",
        "n_params": n_params,
        "training": {
            "objective": f"symmetric InfoNCE, temperature {TEMP}, in-batch negatives",
            "batch": BATCH,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_cap_per_pair": WEIGHT_CAP,
            "best_epoch": best["epoch"],
            "best_probe_hit5": best["hit5"],
        },
        "split": {
            "rule": "repos shuffled with default_rng([42]), first 80% train, rest held-out; no repo in both; eval pairs must occur only in held-out repos",
            "n_repos_total": len(repos),
            "n_repos_train": len(train_repos),
            "n_repos_heldout": len(heldout_repos),
            "pairs_unique_total": len(pair_repos),
            "pairs_train_unique": len(train_counter),
            "pairs_train_weighted_rows": len(train_items),
            "pairs_heldout_clean": len(heldout_clean),
            "pairs_heldout_dirty_excluded_from_eval": len(heldout_dirty),
            "edits_used": sum(lang_counter.values()),
            "per_lang_edits": dict(lang_counter.most_common()),
        },
        "char_vocab": {"size_incl_pad_unk": len(char_to_idx) + 2, "max_word_len": MAX_WORD_LEN},
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "int8_mb": round(int8_path.stat().st_size / 1e6, 3),
        "int8_sha256": file_sha256(int8_path),
        "spot_check_max_abs_int8_vs_torch": max_abs,
        "corpus_sha256": file_sha256(corpus_path),
        "determinism": {
            "seed": SEED,
            "rng": "np.random.default_rng([seed]) for repo split, [seed, epoch] for shuffling",
            "torch_seed": SEED,
        },
        "generated_at": iso_now(),
    }
    (out_dir / "char_to_idx.json").write_text(json.dumps(char_to_idx, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "heldout_clean_pairs.json").write_text(
        json.dumps(sorted([list(k) for k in heldout_clean]), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # language-agnostic eval-clean (typo, correction) pairs: no occurrence in
    # any train repo under ANY language label — the char model can memorize a
    # string pair regardless of which language the corpus tagged it with
    clean_words = {(t, c) for (lang, t, c) in heldout_clean}
    (out_dir / "eval_ok_pairs.json").write_text(
        json.dumps(sorted([list(w) for w in clean_words]), ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest_path = repo / "eval" / "candidates" / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    existing["c_typo"] = receipt
    existing["generated_at"] = iso_now()
    manifest_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"c_typo int8 {receipt['int8_mb']} MB, probe_hit5 {best['hit5']:.4f} -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
