# 06 — Tiered Models: mini / fluency / full

Implements the model-tier pillar of
`kotoshu/TODO.impl/65-universal-kotoshu.md`. Builds on
`04-conversion-pipeline.md`; blocked-by/feeds `07-registry-and-release-distribution.md`.

## Goal

Every manifest language ships in three accuracy/size tiers, each gated
by a measured eval — not by a size target alone:

| Tier | Target | Starting recipe (eval decides) | Role |
|---|---|---|---|
| `mini` | 1–3 MB | top-30k vocab · SVD 300→64 dims · int8 | slim/edge/wasm default |
| `fluency` | 5–15 MB | top-100k vocab · SVD 300→128 dims · int8 | **the default** — Word's 14 MB WordModels.bundle is the existence proof |
| `full` | ~120 MB | current fp32 300d file, unchanged | maximum accuracy, opt-in |

Sizes are design targets, not acceptance gates; the **accuracy gates
below are the acceptance**.

## Why

- 120 MB per language is a non-starter for editors, CI, wasm, and
  metered networks. A 14 MB fluency tier that reranks suggestions
  within ~5% of full quality serves nearly everyone.
- Tier metadata must exist before kotoshu-rs P3 (`EmbeddingProvider`
  + resource parsing) so the Rust core and the Ruby gem resolve the
  same registry shape.

## Method (per language, per tier)

1. **Vocabulary cap.** Keep the top-K words by corpus frequency (Kelly
   tiers where present; else the `.vec` order, which is
   frequency-sorted). Subword/OOV handling degrades with K — the eval
   must measure it, not assume it.
2. **Dimensionality reduction.** Truncated SVD (numpy) 300→{128, 64},
   projection applied to the capped matrix; store the projection only
   if a later rebuild needs it (models stay word-keyed lookups).
3. **Quantization.** int8 per-row (per-embedding) scale — either
   `onnxruntime.quantization.quantize_dynamic` or a hand-rolled
   per-row scale MatMul; whichever verifies byte-exact through the
   eval. Record the method in the manifest.
4. **Artifact naming** (superset of the existing layout, nothing
   moves or is deleted):
   ```
   models/{lang}/fasttext.{lang}.onnx              # full (existing)
   models/{lang}/fasttext.{lang}.vocab.json        # full (existing)
   models/{lang}/fasttext.{lang}.mini.onnx
   models/{lang}/fasttext.{lang}.mini.vocab.json
   models/{lang}/fasttext.{lang}.fluency.onnx
   models/{lang}/fasttext.{lang}.fluency.vocab.json
   ```

## Eval harness (the heart of this plan)

`eval/` in this repo, runnable locally and from CI:

- **Corpus per language**: the gem's suggestion/rerank fixtures
  (exported as conformance vectors — same source as kotoshu-rs tests,
  one source of truth) plus a held-out similarity/relatedness set.
- **Metrics** vs the `full` tier of the same language:
  1. `top1_agreement` — same top-1 suggestion after rerank,
  2. `rank_corr` — Spearman ρ of candidate scores,
  3. `oov_rescue_rate` — OOV words still rescued via remaining vocab.
- **Gates (initial, tunable by the owner with data in hand)**:
  - fluency: `top1_agreement ≥ 0.95`, `rank_corr ≥ 0.97`
  - mini: `top1_agreement ≥ 0.85`, `rank_corr ≥ 0.90`
- **Report**: `eval/reports/{lang}.{tier}.json` committed with the
  model — the manifest links it. A tier that misses its gate does not
  ship; the recipe is adjusted (dims/K/scale) until it passes or the
  tier is dropped for that language and the registry says so.

## Tasks

1. `scripts/build_tiers.py` — one language → all tiers + eval report
   (extends the existing conversion scripts; idempotent).
2. `eval/` harness + gate thresholds as data (`eval/gates.json`).
3. Build tiers for the six full-feature languages first
   (de en es fr pt ru), then the remaining manifest languages
   (ja ko zh) where the corpus allows.
4. Extend `manifest.json` entries with a `tier` block:
   `tier: {name, dims, vocab_size, quantization, eval_ref}` —
   schema co-owned with plan 07 (Resource Spec v1).
5. `.gitignore` stays honest: generated tiers are build artifacts
   until the release in plan 07 uploads them; the repo tracks
   manifests, eval reports, vocab, and metadata — never deletes the
   existing full models (global rule).

## Acceptance

- `scripts/build_tiers.py --lang en` produces both tiers + reports;
  gates pass or the failure is explicit in the report.
- Manifest carries tier blocks for every shipped tier.
- Eval reports are committed and reproducible from a clean checkout +
  upstream `.vec` sources.
- No existing file is modified or deleted; only new tier files added.

## Dependencies

- Needs: upstream `.vec` sources (already used by plan 04's pipeline).
- Feeds: `07-registry-and-release-distribution.md` (registry + first
  release), `kotoshu/TODO.impl/67` M1 (gem tier-aware ModelCache),
  kotoshu-rs P3 (parses the same tier metadata).
- Owner decisions flagged: gate thresholds, default tier, license
  attribution wording.

## Status

_Planning._
