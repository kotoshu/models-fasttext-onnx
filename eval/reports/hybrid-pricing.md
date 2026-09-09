# Plan 115 — Hybrid retrieval pricing: SHIP as opt-in

Question (from the plan-114 reject): the hybrid (C v2 top-20 retrieval +
fastText FULL rescore) was the only variant that beat the full tier top-1
AND top-5 on all four real C-benchmark components — what does its
retrieval half really cost, and does it survive the plan-114 decision
rule scored on REAL pairs only?

Answer: **SHIP as an opt-in registry resource — the first candidate to
clear every clause.** The retrieval matrix prices at 25.8 MB stored int8
per language (102.4 MB fp32, 51.2 MB fp16) but never needs to ship as a
download: the 0.481 MB model derives it at load in ~5 s. int8 per-row
quantization is ranking-identical to the fp32 brute-force sweep on every
real component. On real pairs only (no synth pooling): en +6.3 pp top-5
(CI +4.9 to +7.7, n=2509), de +28.6 pp (+7.1 to +50.0, n=14), ru +11.1 pp
(n=27), es +10.0 pp (n=20) — no component regresses at any k, including
the es slice that rejected C v2. Resource: `kotoshu://models/typo/typo-biencoder`
(models/typo/, 0.481 MB int8 ONNX + char vocab), riding the registry on
the plan-113 additive template: media mirror live on merge, primary null
until the owner cuts a release carrying the assets. No tag here — tags
are the owner decision.

Every number comes from `eval/hybrid_pricing_bench.py` on the frozen
plan-114 C-benchmark (sha256 of every bench file re-verified against the
freeze receipt before scoring), same `corpus_bench` ranking rule as
plans 111/114. Hardware: Apple M1 Max (arm64), CPUExecutionProvider.
Machine receipt: `eval/reports/hybrid-pricing.json`.

## 1. The retrieval matrix, priced (per language, 100k x 256)

| variant | bytes | MB | quantize | load (dequant+renorm) | max abs err | mean 1-cos |
|---------|-------|----|----------|----------------------|-------------|------------|
| fp32 (plan-114 brute force) | 102,400,000 | 102.4 | — | — | — | — |
| fp16 | 51,200,000 | 51.2 | 0.05 s | 0.06 s | 1.2e-4 | ~2e-8 |
| int8 + fp16 row scales | 25,800,000 | 25.8 | 0.05 s | 0.06 s | 1.4e-3 | 2.5e-5 |

Build (fresh encode of the whole vocabulary through the 0.481 MB int8
ONNX encoder, warm session; verified byte-parity with the plan-114
matrices): en 4.8 s, de 5.5 s, ru 5.7 s, es 5.5 s (~18-21k words/s).
fp16 fails the 30 MB cap alone (51.2 MB); int8 fits (25.8 MB + 0.481 MB
model = 26.3 MB), so the verdict binds on int8.

Latency, batch-1 per suggest on real queries (median / p90 ms):

| lang | hybrid, fp32 in RAM | hybrid, int8 in RAM (amortized) | hybrid, int8 RAM-lean (dequant per query) |
|------|--------------------|--------------------------------|-------------------------------------------|
| en | 3.89 / 5.63 | 3.98 / 5.79 | 27.31 / 31.76 |
| de | 3.97 / 6.75 | 4.28 / 5.17 | 27.63 / 31.27 |
| ru | 3.68 / 7.10 | 3.35 / 4.51 | 27.62 / 30.53 |
| es | 3.64 / 4.42 | 3.48 / 4.35 | 25.82 / 30.66 |

RAM-lean (int8 matrix kept as int8, dequantized in 8k-row chunks per
query, no fp32 copy materialized) costs ~7x latency but caps the working
set at 25.8 MB; both paths sit far inside the 100 ms cap.

## 2. Quantized retrieval quality vs the brute-force sweep (REAL components)

int8 and fp16 produce hit@1/5/20 identical to the fp32 brute-force
sweep on every real component of every language (to three decimals).
Slate agreement (correction-in-top-20 vs the fp32 slate): en 2508/2509
(0.9996; the single differing pair is fp32-only and changes no final
hit), de/ru/es 20/20, 27/27, 14/14 (1.0000). The slate itself is
recall-bound at 0.405/0.786/0.704/0.500 — the rescore never sees the
correction beyond the slate, which is priced into every hybrid number
below (miss for every k).

## 3. Hybrid vs full tier — REAL pairs only, with CIs

top-1 / top-5 / top-20 hit rates; paired net top-5 with 95% bootstrap
CIs (2.5/97.5 percentiles, 2000 resamples, plan-114 protocol). int8
configuration (identical to fp32/fp16 everywhere):

| lang | n | full tier | hybrid | paired net top-5 (95% CI) | wins-losses top-5 |
|------|---|-----------|--------|---------------------------|-------------------|
| en | 2509 | 0.103 / 0.207 / 0.288 | 0.119 / 0.270 / 0.405 | **+6.3 pp (+4.9, +7.7)** | 239 - 82 |
| de | 14 | 0.214 / 0.429 / 0.714 | 0.286 / 0.714 / 0.786 | **+28.6 pp (+7.1, +50.0)** | 4 - 0 |
| ru | 27 | 0.074 / 0.370 / 0.519 | 0.111 / 0.481 / 0.704 | +11.1 pp (-11.1, +33.3) | 6 - 3 |
| es | 20 | 0.200 / 0.350 / 0.400 | 0.350 / 0.450 / 0.500 | +10.0 pp (-10.0, +30.0) | 3 - 1 |

top-1 paired nets: en +1.6 (+0.9, +2.3), de +7.1 (+0.0, +21.4), ru +3.7
(-11.1, +18.5), es +15.0 (+0.0, +30.0). top-20 paired nets: en +11.8
(+9.8, +13.8), de +7.1, ru +18.5, es +10.0. Synth components were not
scored in this plan at all — no synth pooling for the verdict, per the
plan (plan-114 had already shown the hybrid winning every synth
component, labeled generator-domain evidence).

## 4. Verdict per the pre-declared rule (plan-114 discipline, binding on int8)

- (a) CLEAR WIN on en AND de: **PASS.** en +6.3 pp, CI [+4.9, +7.7].
  de +28.6 pp, CI [+7.1, +50.0] — zero excluded, with 4 hybrid-only
  wins and 0 full-only wins at n=14 (C v2 failed this same clause at
  [+0.0, +42.9] because its raw ranking conceded full-only wins; the
  fastText rescore is what closes that leak).
- (b) DEGRADES NOWHERE (n >= 20 real): **PASS.** en +6.3 (n=2509),
  ru +11.1 (n=27), es +10.0 (n=20). No real component loses at top-1,
  top-5, or top-20 — including the es slice that rejected C v2 at
  -10.0 pp.
- (c) PRICE: **PASS.** Shipped artifact 0.481 MB int8 (the matrix is
  derived at load, ~5 s per language). Even priced as a stored int8
  download, en totals 26.3 MB <= 30 MB. Median per-suggest latency
  3.4-4.3 ms amortized, 25.8-27.6 ms RAM-lean — both <= 100 ms.

**SHIP** (opt-in): `models/typo/typo.biencoder.onnx` (480,786 B,
int8-dynamic, opset 17) + `typo.biencoder.vocab.json` (char vocab,
1,821 chars) + `typo.json` (ground-truth descriptor, provenance,
verdict receipt). Registry: `kotoshu://models/typo/typo-biencoder`,
schema/build/validate wired, fixture tests added. Consumer recipe:
fetch the resource plus the language full tier; at load encode the
100k vocab (~5 s), keep rows fp32 or int8; per suggest take the C
top-20 slate and rescore by fastText-full cosine to the typo.

## 5. Honest caveats

1. de/ru/es real n are 14/27/20 and the corpus is nearly exhausted
   (212/551/186 unique pairs in total — plan-114 finding 4, unchanged).
   ru and es are wins but not CI-clear wins. The ship rests on the rule
   exactly as pre-declared; a real non-English typo corpus remains the
   highest-value next input, not more architecture.
2. The de clause passes on 14 pairs with a bootstrap lower bound of
   exactly +1/14 — thin by any standard, and reported as such.
3. int8 ranking-identity is measured on this bench, not proven in
   general; a consumer wanting margin can store fp16 at 2x size (also
   ranking-identical here).
4. No engine consumes the resource yet — opt-in by construction,
   `min_engine_version` 0.7 declared, nothing fetches it by default.
5. Registry wiring notes: the typo entry keeps `primary` null until a
   release carries the assets (plan-113 template), which also avoids
   the dead-primary failure mode spotted in the existing lid entry
   (lid primary points at v1.5.0, but lid assets were only ever
   uploaded to v1.4.0 — 404; the media mirror works). The release
   asset glob now carries lid and typo files so the next tag fixes
   both; flipping the primaries stays an owner chore.
