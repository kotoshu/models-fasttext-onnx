#!/usr/bin/env python3
"""Train the plan-114 candidate C v2 typo bi-encoder (non-English signal).

v1 (scripts/train_typo_biencoder.py, bake-off candidate C) trained on the
GitHub Typo Corpus alone, which is 98% English — de/ru/es saw almost no
signal, and de/ru/es clean bench subsets were 8-13 pairs. v2 changes ONE
variable, the training data, keeping the v1 architecture and hyperparameters
identical (char-BiGRU 48d/96d/256d, symmetric InfoNCE, temperature 0.05,
batch 512, 8 epochs, lr 3e-3, occurrence weight cap 8, seed 42):

- real pairs: corpus pairs from the 70% TRAIN repos of the plan-114 repo
  split (scripts/build_cbench.py; v1 used 80%), all corpus languages,
  occurrence-weighted with the v1 cap;
- synth pairs: 8000 keyboard-aware (typo, word) pairs per thin language
  (de ru es) from the repo's own generator (eval/noise.py), drawn from the
  30k most frequent vocab words with typo rejection-sampled into the full
  vocabulary, excluding every held-out-clean string pair.

The probe (best-epoch selection) ranks the correction for the first 1000
sorted REAL en C-benchmark pairs among a fixed 10k en vocab sample — the
probe is now the bench itself (v1 probed on an unsorted set slice).

Artifacts land in eval/candidates/c_typo_v2/ (gitignored); the receipt
(sizes, sha256, data counts, hyperparameters, probe metric) goes to
eval/candidates/manifest.json under c_typo_v2 for eval/cbench_bench.py.

Usage:
  python3 scripts/train_typo_biencoder_v2.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SEED = 42
MAX_WORD_LEN = 24
WEIGHT_CAP = 8

CHAR_DIM = 48
GRU_DIM = 96
OUT_DIM = 256
TEMP = 0.05
BATCH = 512
EPOCHS = 8
LR = 3e-3

TARGET_LANGS = ("en", "de", "es", "ru")
SYNTH_TRAIN_LANGS = ("de", "es", "ru")
PROBE_PAIRS = 1000
PROBE_UNIVERSE = 10_000


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    parser = argparse.ArgumentParser(description="Train the plan-114 typo bi-encoder v2 (candidate C v2)")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    p114 = repo / "eval" / "candidates" / "p114"
    for name in ("train_pairs.json", "clean_str_pairs.json"):
        if not (p114 / name).exists():
            parser.error(f"{p114 / name} missing — run scripts/build_cbench.py first")

    real_train = [
        (l, t, c, n)
        for l, t, c, n in json.loads((p114 / "train_pairs.json").read_text(encoding="utf-8"))
    ]
    synth_train: list[tuple[str, str]] = []
    synth_counts: dict[str, int] = {}
    for lang in SYNTH_TRAIN_LANGS:
        path = p114 / f"synth_train.{lang}.json"
        pairs = [tuple(p) for p in json.loads(path.read_text(encoding="utf-8"))]
        synth_train.extend(pairs)
        synth_counts[lang] = len(pairs)
    clean_pairs = [tuple(p) for p in json.loads((p114 / "clean_str_pairs.json").read_text(encoding="utf-8"))]
    print(f"data: {len(real_train)} real train pairs, {len(synth_train)} synth train pairs {synth_counts}")

    # training rows: real pairs occurrence-weighted (cap), synth pairs weight 1
    train_items: list[tuple[str, str]] = []
    for lang, typo, corr, n in real_train:
        train_items.extend([(typo, corr)] * min(n, WEIGHT_CAP))
    real_weighted = len(train_items)
    train_items.extend(synth_train)
    print(f"rows: real {real_weighted} (weight cap {WEIGHT_CAP}) + synth {len(synth_train)} = {len(train_items)}")

    # char vocabulary: v1 rule — every character of the four target languages'
    # full fastText vocabularies plus all training and clean pair words
    chars: set[str] = set()
    for lang in TARGET_LANGS:
        vocab_path = repo / "models" / lang / f"fasttext.{lang}.vocab.json"
        word_to_idx = json.loads(vocab_path.read_text(encoding="utf-8"))["word_to_idx"]
        for w in word_to_idx:
            chars.update(w[:MAX_WORD_LEN])
    for typo, corr in train_items:
        chars.update(typo[:MAX_WORD_LEN])
        chars.update(corr[:MAX_WORD_LEN])
    for typo, corr in clean_pairs:
        chars.update(typo[:MAX_WORD_LEN])
        chars.update(corr[:MAX_WORD_LEN])
    char_to_idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
    print(f"char vocab: {len(char_to_idx)} + pad + unk")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = CharBiEncoder(len(char_to_idx) + 2)  # pad=0, chars 1..N, unk=N+1

    # probe: the REAL en C-benchmark component is the probe (first 1000 sorted)
    cbench_en = json.loads((repo / "eval" / "cbench" / "cbench.en.json").read_text(encoding="utf-8"))
    probe_pairs = sorted((tuple(p) for p in cbench_en["real"]))[:PROBE_PAIRS]
    en_vocab_path = repo / "models" / "en" / "fasttext.en.vocab.json"
    en_words = list(json.loads(en_vocab_path.read_text(encoding="utf-8"))["word_to_idx"])
    prng = np.random.default_rng([SEED, 7])
    probe_words = [en_words[i] for i in prng.choice(len(en_words), size=PROBE_UNIVERSE, replace=False)]
    probe_universe = probe_words + [c for _, c in probe_pairs]
    probe_universe_enc = torch.from_numpy(encode_words(probe_universe, char_to_idx))
    probe_col: dict[str, int] = {}
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
    train_arr = encode_words([w for pair in train_items for w in pair], char_to_idx).reshape(
        len(train_items), 2, MAX_WORD_LEN
    )
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

    out_dir = repo / "eval" / "candidates" / "c_typo_v2"
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
        raise RuntimeError(f"c_typo_v2 int8 ONNX diverges from torch (max abs {max_abs:.4g})")

    n_params = sum(p.numel() for p in model.parameters())
    receipt = {
        "name": "c_typo_v2",
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
            "probe_note": "hit@5 of the correction among a 10k en vocab universe for the first 1000 sorted REAL en C-benchmark pairs (the probe is the bench)",
        },
        "data": {
            "real_train_unique": len(real_train),
            "real_train_weighted_rows": real_weighted,
            "synth_train": synth_counts,
            "synth_source": "eval/noise.py make_typo, corrections from the 30k most frequent vocab words, typo rejection-sampled into the full vocab",
            "clean_rule": "synth training excludes every held-out-clean string pair (build_cbench exclusion rule)",
        },
        "char_vocab": {"size_incl_pad_unk": len(char_to_idx) + 2, "max_word_len": MAX_WORD_LEN},
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "int8_mb": round(int8_path.stat().st_size / 1e6, 3),
        "int8_sha256": file_sha256(int8_path),
        "spot_check_max_abs_int8_vs_torch": max_abs,
        "determinism": {
            "seed": SEED,
            "rng": "np.random.default_rng([seed, epoch]) for shuffling, [seed, 7] for the probe universe",
            "torch_seed": SEED,
        },
        "generated_at": iso_now(),
    }
    (out_dir / "char_to_idx.json").write_text(json.dumps(char_to_idx, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_path = repo / "eval" / "candidates" / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    existing["c_typo_v2"] = receipt
    existing["generated_at"] = iso_now()
    manifest_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"c_typo_v2 int8 {receipt['int8_mb']} MB, probe_hit5 {best['hit5']:.4f} -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
