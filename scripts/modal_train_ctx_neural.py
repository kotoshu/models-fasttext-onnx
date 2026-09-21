#!/usr/bin/env python3
"""Train the neural context scorer (plan 17 / TODO.deploy/15).

A tiny word-level cloze transformer over the existing 100k full-tier
vocabulary: window of +/-8 context tokens around a gap position,
predict the missing center word. S(c, ctx) = log softmax(logits)[c].
Runs on Modal GPU; exports ONNX at opset 17 / IR version 10.

    python scripts/modal_train_ctx_neural.py --lang en --steps 150000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "pyarrow==17.0.0", "onnx==1.17.0", "onnxruntime==1.20.1")
)

app = modal.App("kotoshu-ctx-neural", image=image)

volume = modal.Volume.from_name("kotoshu-corpus", create_if_missing=True)

WINDOW = 8
D_MODEL = 256
LAYERS = 2
HEADS = 4
FFN = 1024
CTX_LEN = 2 * WINDOW            # context slots (the gap is positional)
LR = 3e-4
WARMUP = 1000


def _clean_token(tok: str) -> str:
    # parity with eval_realword_detection._clean_token
    return tok.strip(".,!?;:()[]{}\"'`´„“”…)([]…—–-").lower()


@app.function(gpu="A10G", timeout=21600, volumes={"/corpus": volume})
def train(lang: str, steps: int, corpora_remote: list, vocab_remote: str,
          d_model: int = 256, layers: int = 2, heads: int = 4, ffn: int = 1024) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    device = "cuda"

    class Cloze(nn.Module):
        """context_ids: [B, CTX_LEN] token ids (PAD = vocab_size);
        pos_ids: [B, CTX_LEN] gap-aware positions 0..CTX_LEN (position
        WINDOW is reserved for the missing center, so left context
        occupies 0..W-1 and right context W+1..2W)."""

        def __init__(self, vocab: int, d_model: int, layers: int, heads: int, ffn: int):
            super().__init__()
            self.tok = nn.Embedding(vocab + 1, d_model)  # +1 = PAD
            self.pos = nn.Embedding(CTX_LEN + 1, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, heads, ffn, batch_first=True, norm_first=True,
                activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers)
            self.head = nn.Linear(d_model, vocab + 1)  # tied rows include PAD; never a target
            self.head.weight = self.tok.weight

        def forward(self, context_ids, pos_ids):
            h = self.tok(context_ids) + self.pos(pos_ids)
            h = self.enc(h)
            return self.head(h.mean(dim=1))

    vocab = json.loads((Path("/corpus") / vocab_remote).read_text())
    vocab = vocab["word_to_idx"] if "word_to_opts" in vocab else vocab
    vocab = vocab["word_to_idx"] if "word_to_idx" in vocab else vocab
    V = max(vocab.values()) + 1
    PAD = V

    shard_i = 0

    def refill(buf: list, need: int):
        nonlocal shard_i
        buf.clear()
        for _attempt in range(len(corpora_remote)):
            fname = corpora_remote[shard_i % len(corpora_remote)]
            shard_i += 1
            pf = pq.ParquetFile(str(Path("/corpus") / fname))
            for batch in pf.iter_batches(batch_size=256, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    # sentence-level windows (the eval's own unit):
                    # eval sentences are short and PAD-heavy at the
                    # edges, so training splits into sentences and pads
                    # identically. Every in-vocab center.
                    for sent in re.split(r"[.!?]+\s*", (text or "")):
                        ids = [vocab[t] for t in (_clean_token(w) for w in sent.split())
                               if t in vocab]
                        if len(ids) < 3:
                            continue
                        for i in range(len(ids)):
                            if ids[i] == 0:
                                continue
                            left = ids[max(0, i - WINDOW):i]
                            right = ids[i + 1:i + 1 + WINDOW]
                            buf.append((left, right, ids[i]))
                            if len(buf) >= need:
                                return

    model = Cloze(V, d_model, layers, heads, ffn).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def lr_at(s):
        if s < WARMUP:
            return (s + 1) / WARMUP
        return 0.5 * (1 + math.cos(math.pi * s / steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    buf: list = []
    t0 = time.time()
    losses: list[float] = []
    for step in range(1, steps + 1):
        if len(buf) < 512:
            refill(buf, 150000)
        xs = torch.full((256, CTX_LEN), PAD, dtype=torch.long, device=device)
        ys = torch.zeros(256, dtype=torch.long, device=device)
        for b in range(256):
            left, right, target = buf.pop()
            xs[b, :len(left)] = torch.tensor(left, device=device)
            xs[b, WINDOW:WINDOW + len(right)] = torch.tensor(right, device=device)
            ys[b] = target
        # gap-aware positions: left 0..W-1, right W+1..2W (truncated ctx
        # is left-aligned here; eval pads the same way, so the
        # distributions match)
        pos = torch.cat([torch.arange(WINDOW), torch.arange(WINDOW + 1, CTX_LEN + 1)])
        pos = pos.to(device).unsqueeze(0).expand(256, -1).contiguous()
        logits = model(xs, pos)
        loss = F.cross_entropy(logits, ys)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 1000 == 0:
            print(f"step {step}/{steps} loss {sum(losses[-1000:]) / min(1000, step):.4f} "
                  f"({(time.time() - t0) / 60:.1f} min, {n_params / 1e6:.1f}M params)",
                  flush=True)

    # checkpoint first: an export bug must never cost the training run
    torch.save({"sd": model.state_dict(), "vocab_size": V}, f"/corpus/{lang}.ctx-neural.pt")
    volume.commit()

    # ---- export: ONNX opset 17, IR 10 --------------------------------
    model.eval()
    model.to("cpu")  # export and parity run on CPU tensors
    # the fused _transformer_encoder_layer_fwd kernel is not ONNX-exportable
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except AttributeError:
        pass
    example_x = torch.full((1, CTX_LEN), PAD, dtype=torch.long)
    example_p = torch.cat([torch.arange(WINDOW),
                           torch.arange(WINDOW + 1, CTX_LEN + 1)]).unsqueeze(0)
    # fp16 export: fp32 is 109 MB (over the 100 MiB git limit), and
    # dynamic int8 on nn.Embedding asserts without a special qconfig;
    # fp16 halves to ~55 MB and runs everywhere including wasm.
    model.half()
    example_x = example_x.to(torch.int64)
    out_path = f"/corpus/{lang}.ctx-neural.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (example_x, example_p), out_path,
            input_names=["context_ids", "pos_ids"], output_names=["logits"],
            dynamic_axes={"context_ids": {0: "batch"},
                          "pos_ids": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17)
    m = onnx.load(out_path)
    m.ir_version = 10
    onnx.save(m, out_path)

    with torch.no_grad():
        ref = model(example_x, example_p).numpy()
    got = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"]).run(
        None, {"context_ids": example_x.numpy(), "pos_ids": example_p.numpy()})[0]
    parity_err = float(np.abs(ref.astype(np.float32) - got.astype(np.float32)).max())

    volume.commit()
    return {"params_m": round(n_params / 1e6, 1),
            "final_loss": round(sum(losses[-1000:]) / min(1000, steps), 4),
            "onnx_bytes": Path(out_path).stat().st_size,
            "parity_err": parity_err, "minutes": round((time.time() - t0) / 60, 1)}


EN_SHARDS = [
    "eval/corpus/en-wiki-train-00000.parquet",
    "eval/corpus/en-train-00001.parquet",
    "eval/corpus/en-train-00002.parquet",
]


@app.local_entrypoint()
def main(lang: str = "en", steps: int = 150000, shards: int = 1,
         d_model: int = 256, layers: int = 2, heads: int = 4, ffn: int = 1024):
    vocab_local = REPO_ROOT / f"models/{lang}/fasttext.{lang}.vocab.json"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    corpus_remotes = []
    with volume.batch_upload() as batch:
        for i in range(shards):
            local = EN_SHARDS[i]
            assert (REPO_ROOT / local).exists(), f"missing shard {local}"
            remote = f"neural-{lang}-{stamp}-s{i}.parquet"
            batch.put_file(str(REPO_ROOT / local), remote)
            corpus_remotes.append(remote)
        vocab_remote = f"neural-{lang}-{stamp}.vocab.json"
        batch.put_file(str(vocab_local), vocab_remote)
    result = train.remote(lang, steps, corpus_remotes, vocab_remote,
                          d_model=d_model, layers=layers, heads=heads, ffn=ffn)
    print(json.dumps(result, indent=1))
