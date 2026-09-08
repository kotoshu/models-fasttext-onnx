#!/usr/bin/env python3
"""Prepare plan-111 bakeoff candidate weights (B rerank, D ModernBERT).

Downloads HF weights OUTSIDE git (eval/candidates/, gitignored), exports
ONNX fp32 via torch.onnx.export, and quantizes to int8 dynamic weights
(onnxruntime.quantization.quantize_dynamic — the same int8 class the
shipped tiers use). Nothing here is committed: the committed evidence is
the size/sha256 receipt embedded in eval/reports/bakeoff.*.json by
eval/bakeoff_bench.py and a spot-check parity between torch fp32 and the
quantized ONNX run by this script.

Candidates:
- B  cross-encoder/ms-marco-MiniLM-L-6-v2 as a (context, candidate) rerank
   scorer: BertForSequenceClassification with 1 logit, inputs
   input_ids/attention_mask/token_type_ids, output logits [B, 1].
- D  answerdotai/ModernBERT-small and ModernBERT-base as zero-shot bi-encoders:
   mean-pooled last hidden state over the attention mask, L2-normalized inside
   the graph so cosine is a dot product downstream. base is ~149 MB at int8
   (over the 30 MB budget); small is ~30 MB (inside). Both are prepared so the
   report can show whether the class is worth a purpose-trained follow-up.

Usage:
  python3 scripts/prepare_bakeoff_candidates.py --repo-root . [--only b d_small d_base]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

OPSET = 17
CAND_DIR_DEFAULT = "eval/candidates"
SPOT_TOL = 0.05  # int8 vs torch on L2-normalized embeddings (range [-1, 1])
SPOT_TOL_LOGIT = 0.5  # int8 vs torch on unbounded relevance logits (|logit| ~ 10)

CANDIDATES = {
    "b_rerank": {
        "hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "kind": "rerank",
        "status": "available",
        "note": "B: fastText retrieval + cross-encoder rerank over (context, candidate)",
    },
    "d_modernbert_small": {
        "hf_id": "answerdotai/ModernBERT-small",
        "kind": "biencoder",
        "status": "not_found_on_hub",
        "note": (
            "requested in-budget ModernBERT-small (~30M params, ~30 MB int8) does "
            "not exist: answerdotai publishes only base/large (HF API author "
            "listing + search, 2026-09-08). Recorded as unavailable, not "
            "evaluated — same discipline as the model2vec potion-mini slot."
        ),
    },
    "d_modernbert_base": {
        "hf_id": "answerdotai/ModernBERT-base",
        "kind": "biencoder",
        "status": "available",
        "note": "D candidate (only released encoder size under large; 149M params, int8 ~149 MB > 30 MB cap — over budget, measured anyway to price the class)",
    },
}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RerankWrapper(torch.nn.Module):
    """(input_ids, attention_mask, token_type_ids) -> relevance logit [B, 1]."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return out.logits


class BiEncoderWrapper(torch.nn.Module):
    """(input_ids, attention_mask) -> L2-normalized mean-pooled embedding [B, H]."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return torch.nn.functional.normalize(pooled, dim=-1)


def export_torch(module: torch.nn.Module, dest: Path, sample_inputs: dict, input_names: list[str]) -> None:
    args = tuple(sample_inputs[n] for n in input_names)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            args,
            str(dest),
            input_names=input_names,
            output_names=["logits"] if len(input_names) == 3 else ["embedding"],
            dynamic_axes={n: {0: "batch", 1: "seq"} for n in input_names},
            opset_version=OPSET,
            dynamo=False,
        )


def load_model(hf_id: str, arch) -> tuple[object, list[str], list[str]]:
    """Load weights bypassing transformers.from_pretrained.

    On this stack (transformers 5.14 + torch 2.12, arm64) the from_pretrained
    lazy-load path leaves BERT weights in a state whose first Linear touch
    SIGBUSes or yields NaN logits (reproduced 2026-09-08; the hub blob itself
    sha256-verifies and the same tensors load cleanly via safetensors).
    Loading the safetensors blob into a config-built model works and matches
    the ONNX export, so we do that instead. Missing/unexpected keys are
    returned for the receipt so a silent mis-load cannot hide.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg = AutoConfig.from_pretrained(hf_id)
    model = arch.from_config(cfg)
    blob = hf_hub_download(repo_id=hf_id, filename="model.safetensors")
    state = load_file(blob)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, list(missing), list(unexpected)


def prepare(name: str, cand: dict, repo: Path, cand_root: Path) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir = cand_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    hf_id = cand["hf_id"]
    print(f"[{name}] downloading {hf_id} (HF cache; weights stay outside git)")
    fp32_path = out_dir / "model.fp32.onnx"
    int8_path = out_dir / "model.int8.onnx"

    if cand["kind"] == "rerank":
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model, missing, unexpected = load_model(hf_id, AutoModelForSequenceClassification)
        wrapper = RerankWrapper(model)
        enc = tokenizer(
            ["a spelling mistake in context", "another one"],
            ["candidate word one", "candidate word two"],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        sample = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids")}
        input_names = list(sample)
        # second, differently-shaped validation batch (different content,
        # different padding) — catches a graph whose mask handling was baked
        # to the export batch
        enc2 = tokenizer(
            ["short ctx", "a much longer context sentence with several more tokens than the first row"],
            ["x", "candidate"],
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt",
        )
        spot2 = {k: enc2[k] for k in input_names}
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model, missing, unexpected = load_model(hf_id, AutoModel)
        wrapper = BiEncoderWrapper(model)
        enc = tokenizer(["misspell", "correct"], padding="max_length", max_length=16, truncation=True, return_tensors="pt")
        sample = {k: enc[k] for k in ("input_ids", "attention_mask")}
        input_names = list(sample)
        enc2 = tokenizer(["kalibr", "orthographe"], padding=True, truncation=True, max_length=24, return_tensors="pt")
        spot2 = {k: enc2[k] for k in input_names}

    # torch references FIRST, before any tracing: the legacy torch.onnx.export
    # tracer mutates model state (transformers 5.x attention-mask cache), so a
    # post-export torch forward is no longer a trustworthy reference — verified
    # 2026-09-08: post-export forward moves logits by ~0.8 while the exported
    # graph matches the pristine forward to 4e-6.
    with torch.inference_mode():
        ref1 = wrapper(**sample).numpy()
        ref2 = wrapper(**spot2).numpy()

    print(f"[{name}] exporting fp32 ONNX (opset {OPSET})")
    export_torch(wrapper, fp32_path, sample, input_names)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    tokenizer.save_pretrained(str(out_dir))

    # spot-check both batches: torch fp32 (pristine) vs fp32 ONNX vs int8 ONNX;
    # NaN anywhere is a hard failure (a NaN comparison would otherwise pass
    # every `< tol` check silently — seen once, never again).
    import onnxruntime as ort

    fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    fp32_max_abs = 0.0
    int8_max_abs = 0.0
    for batch in (sample, spot2):
        feeds = {n: v.numpy() for n, v in batch.items()}
        fp32_out = fp32_sess.run(None, feeds)[0]
        int8_out = int8_sess.run(None, feeds)[0]
        ref = ref1 if batch is sample else ref2
        for label, arr in (("torch", ref), ("onnx_fp32", fp32_out), ("onnx_int8", int8_out)):
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"{name}: {label} output is not finite: {arr.ravel()[:4]}")
        fp32_max_abs = max(fp32_max_abs, float(np.abs(fp32_out - ref).max()))
        int8_max_abs = max(int8_max_abs, float(np.abs(int8_out - ref).max()))
    int8_tol = SPOT_TOL_LOGIT if cand["kind"] == "rerank" else SPOT_TOL
    print(f"[{name}] spot vs pristine torch (2 batches): fp32 {fp32_max_abs:.2e} (tol 1e-3), int8 {int8_max_abs:.4f} (tol {int8_tol})")
    if fp32_max_abs > 1e-3:
        raise RuntimeError(f"{name}: fp32 ONNX diverges from torch (max abs {fp32_max_abs:.3g})")
    if int8_max_abs > int8_tol:
        raise RuntimeError(f"{name}: int8 ONNX diverges from torch (max abs {int8_max_abs:.4g} > {int8_tol})")

    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = int8_path.stat().st_size / 1e6
    n_params = sum(p.numel() for p in wrapper.model.parameters())
    receipt = {
        "name": name,
        "hf_id": hf_id,
        "kind": cand["kind"],
        "note": cand["note"],
        "n_params": n_params,
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "int8_mb": round(int8_mb, 2),
        "int8_sha256": file_sha256(int8_path),
        "opset": OPSET,
        "quantization": "onnxruntime quantize_dynamic (QInt8 weights)",
        "load_missing_keys": missing,
        "load_unexpected_keys": unexpected,
        "spot_check_max_abs_fp32_onnx_vs_torch": fp32_max_abs,
        "spot_check_max_abs_int8_onnx_vs_torch": int8_max_abs,
        "spot_check_tol": SPOT_TOL_LOGIT if cand["kind"] == "rerank" else SPOT_TOL,
        "spot_check_batches": 2,
        "spot_check_input_shape": {k: list(v.shape) for k, v in sample.items()},
        "loader_note": (
            "weights loaded via safetensors load_file into a config-built model "
            "(from_pretrained lazy-load path SIGBUSes/NaNs on transformers 5.14 + "
            "torch 2.12 arm64; see load_model docstring)"
        ),
        "reference_note": (
            "torch references captured before export: the legacy tracer mutates "
            "the transformers 5.x attention-mask cache, so post-export torch "
            "forwards are untrustworthy"
        ),
    }
    print(f"[{name}] int8 {receipt['int8_mb']} MB, sha256 {receipt['int8_sha256'][:16]}…")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare plan-111 bakeoff candidate weights")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    parser.add_argument("--only", nargs="+", choices=sorted(CANDIDATES), help="prepare only these candidates")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    cand_root = repo / CAND_DIR_DEFAULT
    cand_root.mkdir(parents=True, exist_ok=True)

    todo = {k: v for k, v in CANDIDATES.items() if args.only is None or k in args.only}
    torch.manual_seed(42)
    receipts = {}
    for name, cand in todo.items():
        if cand.get("status") != "available":
            receipts[name] = {"name": name, "hf_id": cand["hf_id"], "status": cand["status"], "note": cand["note"]}
            print(f"[{name}] skipping: {cand['status']}")
            continue
        receipts[name] = prepare(name, cand, repo, cand_root)

    manifest_path = cand_root / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    existing.update(receipts)
    existing["generated_at"] = iso_now()
    manifest_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
