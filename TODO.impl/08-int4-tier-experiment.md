# 08 — int4 tier experiment (plan 101, eval-gated)

Branch copy of `kotoshu/TODO.impl/101-int4-tier-experiment.md` (the
specification) plus the measured verdict. The experiment runs here, in
the models repo, against the unchanged eval harness.

## Why
The tier recipe settled at int8 (rank_corr 0.9999, near-lossless) after
SVD failed its gates. int4 halves the bytes again — mini ~1.5 MB,
fluency ~8 MB — which changes browser adoption math for the wasm
playground. It was named in the original kotoshu-rs TODO (B1) but never
run through the gates with per-row scales.

Prior int4 work (plan 68 B1, eval/reports/int4*.json): group-wise scales
(128/64/32 elements per fp32 scale) over the full vocab, both rounding
modes — all rejected on the fluency top1 gate. The one untried lever was
exactly what plan 101 asks for: per-row scales (the kotoshu-rs
RowFormat 0x04 shape) over the tier vocab cuts.

## What was run
`scripts/build_int4_per_row.py`: packed signed nibbles (element 2j in
the HIGH nibble of byte j — the kotoshu-rs `pack_row_int4` contract) +
ONE fp16 scale per row (`max_abs(row)/7`, codes rounded nearest-even,
quantized against the fp16-rounded scale so dequant is exact), over the
SAME vocab cuts as the shipped int8 tiers (mini top-10k, fluency top-50k
— sizes read from `models/{lang}/tiers.json`). Languages: en de es fr ru
pt (the original six) + it pl from the newer set. Gates unchanged:
`run_eval.evaluate` with the existing thresholds, and the same seeded
probe stream + identical vocab as the int8 runs, so the deltas are
apples-to-apples. Acceptance per plan 101: rank_corr within 0.001 of the
int8 tier AND top1 gate holds.

## Measured verdict (2026-09-07): REJECTED

16/16 language/tier candidates miss the acceptance. rank_corr deltas vs
int8 are -0.0146 to -0.0199 (15-20x outside the 0.001 window); top1
agreement fails its gate on all 8 fluency4 (0.708-0.923 vs >= 0.95) and
2 of 8 mini4 (de 0.800, es 0.810 vs >= 0.85). Only 6/16 pass even the
plain tier gates. Numbers:
`eval/reports/{lang}.{mini4,fluency4}.json`, cross-language verdict in
`eval/reports/int4-per-row.summary.json`.

| lang | tier | rank_corr (int4 / int8) | top1 (int4 / int8) | gate | verdict |
|---|---|---|---|---|---|
| en | mini4    | 0.9836 / 0.9999 | 1.000 / 1.000 (n=12) | pass | reject (rank_corr) |
| en | fluency4 | 0.9827 / 0.9999 | 0.846 / 1.000 (n=26) | fail | reject |
| de | mini4    | 0.9804 / 0.9999 | 0.800 / 1.000 (n=15) | fail | reject |
| de | fluency4 | 0.9800 / 0.9999 | 0.800 / 1.000 (n=25) | fail | reject |
| es | mini4    | 0.9850 / 0.9999 | 0.810 / 0.952 (n=21) | fail | reject |
| es | fluency4 | 0.9844 / 0.9999 | 0.867 / 1.000 (n=15) | fail | reject |
| fr | mini4    | 0.9840 / 0.9999 | 0.857 / 1.000 (n=14) | pass | reject (rank_corr) |
| fr | fluency4 | 0.9833 / 0.9999 | 0.708 / 1.000 (n=24) | fail | reject |
| ru | mini4    | 0.9810 / 0.9999 | 1.000 / 1.000 (n=8)  | pass | reject (rank_corr) |
| ru | fluency4 | 0.9811 / 0.9999 | 0.824 / 1.000 (n=17) | fail | reject |
| pt | mini4    | 0.9853 / 0.9999 | 0.882 / 1.000 (n=17) | pass | reject (rank_corr) |
| pt | fluency4 | 0.9850 / 0.9999 | 0.810 / 1.000 (n=21) | fail | reject |
| it | mini4    | 0.9838 / 0.9999 | 1.000 / 1.000 (n=9)  | pass | reject (rank_corr) |
| it | fluency4 | 0.9840 / 0.9999 | 0.813 / 1.000 (n=16) | fail | reject |
| pl | mini4    | 0.9819 / 0.9999 | 0.917 / 1.000 (n=12) | pass | reject (rank_corr) |
| pl | fluency4 | 0.9818 / 0.9999 | 0.923 / 1.000 (n=13) | fail | reject |

### Size audit (actual bytes)
- mini4: 1,522,500 B (~1.52 MB) vs int8 mini 3,040,752 B (~3.04 MB);
  vocab json 0.20-0.27 MB.
- fluency4: 7,602,510 B (~7.60 MB) vs int8 fluency 15,200,760 B
  (~15.20 MB); vocab json 1.08-1.50 MB (ru widest).
- Sizes are uniform across the 8 languages (same 100k vocab / 300 dims
  upstream matrix), so the halving is real — the accuracy is not.

### Why this closes the question
Per-row is the COARSEST scale granularity (one scale per 300-d row); the
prior sweep measured finer groupings (g128/g64/g32 fp32 scales, both
roundings, full vocab) at rank_corr 0.983-0.991 with the same top1
failures. The granularity ladder is now measured end to end: no scale
granularity between per-row and group-32 brings 4-bit within 0.001 of
int8, and the vocab cut does not change that (int8 on the same cuts
scores 0.9999/1.0). The 4-bit noise floor (1/14 of row max_abs vs 1/254
for int8) keeps flipping 1-2 typo probes per language. Plan 101 outcome:
a reject on data is a complete outcome (the SVD precedent). int8-per-row
stands as final; no registry, manifest, tiers.json or release-workflow
changes.

## Status
Rejected on measurement — 2026-09-07. Artifacts local-only (gitignored),
reports committed.
