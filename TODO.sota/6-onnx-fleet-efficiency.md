# Plan S6: ONNX fleet efficiency — KD-QAT int8, tiered latency lanes

## Status

researched and scoped (2026-09-22). The efficiency literature converged: KD-QAT (Kur et al. 2025; signal-enhanced KD-QAT arXiv Mar 2024; Punching Above Precision, OpenReview 2025) — quantization-aware distillation beats post-training quantization; production proof-points: e-commerce spell correction with BART/T5 + quantized deployment (Dutta et al., ACM 2024), lightweight transformers at ~19.5ms INT8 latency (arXiv 2026), Gboard's decoder architecture (arXiv:2410.15575), and the client-server split for neural keyboard correction (Mogalle 2025) with server-side LLM lanes (Proofread, arXiv June 2024). Our house law (IR-10 floor, zero-LFS, releases for server artifacts) is the substrate.

## Problem

Every plan in this namespace adds a runtime model. Without a unified efficiency pipeline, per-model ad-hoc quantization diverges and the browser/mini tier drifts from the server tier.

## What

1. One pipeline: train (fp16) → KD into int8 (QAT, teacher = fp16 self) → ONNX IR-10 export → per-tier manifest. All S2/S3/typo/semantic models go through it; no model ships fp32.
2. Latency lanes with budgets: browser-mini (≤5MB, ≤10ms/editor keystroke), editor-fluency (≤20MB, ≤25ms), server-full (≤100ms/sentence, batched). Each lane's manifest carries measured latencies (the harness records them, not vibes).
3. Client-server protocol for lanes above mini (Mogalle pattern): local detect → server rerank, with offline fallback to the pure-Ruby engine (the accelerator law, restated for neural lanes).
4. Conformance: every quantized artifact replays the frozen vectors (byte-identical law extends to the neural lanes via the C1 gate, not vector equality).

## Consumers

S2/S3/S4 models; the browser demo; editor integrations; the fleet's memory law.
