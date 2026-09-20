# Plan D15 (plan 17 authorized): the neural context scorer — train, gate, deploy

## Status

executed with a diagnosed failed-gate (2026-09-20) — v1 trained end to end (27.3M params, 150k steps, loss 0.014, 52 MB fp16 IR-10 artifact, torch/onnx parity) and the gate FAILED on frozen evidence (FP 1% -> recall 0.1%): the margin scale is dominated by UNTRAINED PAD embeddings (training used full windows only; short eval contexts are out-of-distribution — clean text scores +33 nats against real errors' +54). The failure is CONCRETE and fixable — unlike the n-gram's identical distributions — v2 = pad-aware training (padded windows or an attention-mask input), same gate. First run's 86 minutes were lost to an Embedding-quantization export assertion; the trainer now checkpoints before export (a law). En route: Modal 1.5 requires python -m modal run (direct invocation silently no-ops); fp16 export replaces fragile int8 dyn-quant; the eval harness tolerates the absent release-only full-tier onnx.

## Problem

The n-gram rung failed distributionally (clean and error margins are
the same distribution; at FP 1%, recall 5.1%). The design record's
next rung: a small masked-LM over word-in-context — S(c, ctx) =
log P(candidate | ±8 context tokens) — a genuinely different model
class. The gate is unchanged and will not be weakened: eligible-token
FP <= 1% AND covered-instance true-top recall >= 60% on the frozen
eval/realword/en.json split.

## What

1. **Trainer** (`scripts/modal_train_ctx_neural.py`): word-level
   cloze model over the existing 100k full-tier vocab (zero new
   tokenization — confusion tables and eval ids stay aligned), window
   ±8 around a masked center, tiny transformer (d=256, 2 layers, 4
   heads, ~27M params incl. tied embeddings), trained on the on-disk
   en wiki shard via Modal GPU (the proven substrate pattern).
2. **Export**: ONNX at opset 17 / **IR version 10** (the standing
   floor), dynamic sequence length, int8 dynamic quantization —
   target artifact <= 40 MB, browser-loadable via onnxruntime-web.
3. **Gate adapter**: a `neural` scorer in the eval harness protocol
   (same frozen split, same metrics, same FP anchoring as the ctxlm
   run — no protocol drift between runs).
4. **The gate runs and the evidence freezes** — pass or fail, the
   number is the number.
5. **If the gate passes**: the demo gains calibrated real-word
   detection served from the raw host (the transformer in the browser
   — the ecosystem's ultimate proof); plans 146/07 unblock. If it
   fails: the ladder's verdict records it and no threshold moves.
6. Documentation and rules updated (the new artifact class joins the
   storage law's browser set on pass).

## Consumers

gem plan 146, rs plan 07, the demo's detection mode, every future
language (the recipe scales per language only after en passes).
