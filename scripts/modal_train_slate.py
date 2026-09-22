#!/usr/bin/env python3
"""S2: the constrained slate reranker (TODO.sota/2).

A compact cross-encoder over (typo, candidate) pairs: encode each
slate candidate against the typo, softmax over the slate — the model
can ONLY rerank the production SymSpell slate, so out-of-slate output
is 0 by construction (TODO.sota/0 G-S2). Char-level input, d_model
256, 4 layers. The head learns a delta on the SymSpell rank signal
(a learnable linear of the normalized rank) rather than re-deriving
the ordering from characters alone — the char-only variant lost 18pp
to the frequency prior (first honest negative, 2026-09-22).

    python3 -m modal run scripts/modal_train_slate.py --lang en

Upload inputs to the kotoshu-corpus volume first:
    modal volume put kotoshu-corpus en.slates.jsonl /en.slates.jsonl
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "onnx==1.17.0", "onnxruntime==1.20.1", "numpy<2")
)

app = modal.App("kotoshu-slate", image=image)
volume = modal.Volume.from_name("kotoshu-corpus", create_if_missing=True)

TYPO_MAX = 24
CAND_MAX = 24
K = 10
D_MODEL = 256
LAYERS = 4
HEADS = 4
FFN = 512
BATCH = 256
EPOCHS = 30
LR = 3e-4


def build_vocab(rows):
    chars = set()
    for r in rows:
        chars.update(r["typo"].lower())
        for c in r["slate"]:
            chars.update(c.lower())
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 = pad
    vocab["</>"] = len(vocab) + 1  # boundary / unknown
    return vocab


def encode(text, vocab, maxlen):
    unk = vocab["</>"]
    body = [vocab.get(c, unk) for c in text.lower()[: maxlen - 2]]
    ids = [unk] + body + [unk]
    return ids + [0] * (maxlen - len(ids))


@app.function(gpu="A10G", timeout=14400, volumes={"/corpus": volume})
def train(lang: str, epochs: int = EPOCHS):
    import torch

    class SlateReranker(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=HEADS, dim_feedforward=FFN,
                batch_first=True, dropout=0.1)
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=LAYERS)
            self.head = torch.nn.Linear(D_MODEL, 1)
            self.rank_emb = torch.nn.Embedding(K, D_MODEL)

        def forward(self, typo_ids, cand_ids, rank_feat):
            t_mask = typo_ids == 0
            t = self.encoder(self.emb(typo_ids), src_key_padding_mask=t_mask)
            t = t[:, 0]
            B, K, C = cand_ids.shape
            c_mask = cand_ids == 0
            c = self.encoder(
                self.emb(cand_ids.reshape(B * K, C)),
                src_key_padding_mask=c_mask.reshape(B * K, C))
            c = c[:, 0].reshape(B, K, -1)
            rank_ids = (rank_feat * (K - 1)).round().long().clamp(0, K - 1)
            c = c + self.rank_emb(rank_ids)
            return self.head(torch.nn.functional.relu(c + t.unsqueeze(1))).squeeze(-1)

    rows = [json.loads(l) for l in open(f"/corpus/{lang}.slates.jsonl")]
    vocab = build_vocab(rows)
    print(f"rows={len(rows)} vocab={len(vocab)+1}")

    data = []
    for r in rows:
        if len(r["slate"]) < 2:
            continue
        slate = (r["slate"] + [""] * K)[:K]
        ranks = [-(i / max(1, K - 1)) for i in range(K)]
        data.append((r["typo"], slate, r["label"], ranks))

    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(data), generator=rng).tolist()
    n_val = max(1, len(data) // 20)
    val = [data[i] for i in perm[:n_val]]
    trn = [data[i] for i in perm[n_val:]]

    model = SlateReranker(len(vocab) + 1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(trn), generator=rng).tolist()
        t0 = time.time()
        tot = cnt = 0
        for bi in range(0, len(order), BATCH):
            batch = [trn[i] for i in order[bi:bi + BATCH]]
            typos = torch.tensor([encode(t, vocab, TYPO_MAX) for t, _, _, _ in batch]).cuda()
            cands = torch.tensor([[encode(c, vocab, CAND_MAX) for c in slate] for _, slate, _, _ in batch]).cuda()
            ranks = torch.tensor([rk for _, _, _, rk in batch], dtype=torch.float).cuda()
            labels = torch.tensor([max(0, l) if l >= 0 else 0 for _, _, l, _ in batch]).cuda()
            mask = torch.tensor([l >= 0 for _, _, l, _ in batch]).cuda()
            scores = model(typos, cands, ranks)
            loss = torch.nn.functional.cross_entropy(scores, labels, reduction="none")
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        print(f"epoch {epoch}: loss {tot/max(1,cnt):.4f} ({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    hits = total = sym_hits = 0
    with torch.no_grad():
        for bi in range(0, len(val), BATCH):
            batch = val[bi:bi + BATCH]
            typos = torch.tensor([encode(t, vocab, TYPO_MAX) for t, _, _, _ in batch]).cuda()
            cands = torch.tensor([[encode(c, vocab, CAND_MAX) for c in slate] for _, slate, _, _ in batch]).cuda()
            ranks = torch.tensor([rk for _, _, _, rk in batch], dtype=torch.float).cuda()
            scores = model(typos, cands, ranks).argmax(dim=1).cpu()
            for (_, slate, label, _), pred in zip(batch, scores.tolist()):
                if label < 0:
                    continue
                total += 1
                hits += 1 if pred == label else 0
                sym_hits += 1 if label == 0 else 0
    print(f"val: rerank-top1 {hits/total:.4f} vs sym-top1 {sym_hits/total:.4f} (n={total})", flush=True)

    out_dir = Path("/corpus") / lang / "slate"
    out_dir.mkdir(parents=True, exist_ok=True)
    dummy_t = torch.zeros(1, TYPO_MAX, dtype=torch.long)
    dummy_c = torch.zeros(1, K, CAND_MAX, dtype=torch.long)
    dummy_r = torch.zeros(1, K, dtype=torch.float)
    torch.onnx.export(
        model.cpu().eval(), (dummy_t, dummy_c, dummy_r), str(out_dir / "slate.rerank.onnx"),
        input_names=["typo_ids", "cand_ids", "rank_feat"], output_names=["scores"],
        dynamic_axes={"typo_ids": {0: "B"}, "cand_ids": {0: "B"},
                      "rank_feat": {0: "B"}, "scores": {0: "B"}},
        opset_version=17)
    (out_dir / "slate.vocab.json").write_text(json.dumps(vocab))
    (out_dir / "slate.config.json").write_text(json.dumps(
        {"lang": lang, "typo_max": TYPO_MAX, "cand_max": CAND_MAX, "k": K,
         "d_model": D_MODEL, "layers": LAYERS, "heads": HEADS, "epochs": epochs,
         "val_rerank_top1": hits / total, "val_sym_top1": sym_hits / total}))
    volume.commit()
    print(f"exported {out_dir/'slate.rerank.onnx'}")


@app.local_entrypoint()
def main(lang: str = "en", epochs: int = EPOCHS):
    train.remote(lang, epochs)
