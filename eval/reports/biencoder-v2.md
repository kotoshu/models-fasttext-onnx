# Plan 114 — Typo bi-encoder v2: honest reject on the es real gate

Question (from the bake-off follow-up): with non-English training signal, a
frozen 2000+ pair repo-clean bench, and the same decision discipline as
int4 — does the purpose-trained char-BiGRU typo bi-encoder (candidate C)
ship as an opt-in registry resource?

Answer: **REJECT — no registry change, no tag, the fastText tiers stay the
shipped models.** C v2 wins English clearly on 2509 genuinely-unseen real
pairs and wins German on every component measured, but it regresses real
Spanish top-5 by 10 pp on the only n>=20 Spanish real slice — a breach of
the pre-declared "degrades nowhere" gate. Three independent Spanish real
slices (bake-off 13 pairs, v1-here 13, v2-here 20) all point the same way.
A positive thread survives (the hybrid), recorded below for a future plan,
not shipped here.

Every number comes from `eval/cbench_bench.py` on the frozen plan-114
C-benchmark (`scripts/build_cbench.py`, receipt
`eval/reports/cbench.frozen.json`) — the exact `corpus_bench` ranking rule
(rank the human correction among the whole vocabulary by cosine to the typo,
typo excluded, ties optimistic). Hardware: Apple M1 Max (arm64),
CPUExecutionProvider. Candidate receipts: `eval/candidates/manifest.json`
(`c_typo`, `c_typo_v2`).

## The bench (frozen)

| lang | real pairs | synth pairs | real construction |
|------|-----------|-------------|-------------------|
| en | 2509 | 0 | held-out-repo-clean corpus pairs, both words in the 100k full vocab, no sampling cap, 70/30 repo split (seed 42) |
| de | 14 | 2000 | same rule; corpus has only 212 unique de pairs total |
| ru | 27 | 2000 | same rule |
| es | 20 | 2000 | same rule |

A pair is clean iff it occurs in NO train repo under ANY language label
(v1 string-level rule). Synth pairs (de/ru/es only — en real alone exceeds
the 2000 target) are `eval/noise.py` keyboard-model typos of the 30k most
frequent vocab words, rejection-sampled so typo and correction are both in
the full vocabulary, disjoint from every training source (real train, synth
train, any language label). Seeds: train `default_rng([42, crc32(lang), 1140])`,
bench `default_rng([42, crc32(lang), 1141])`.

## The models

| model | int8 MB | training data |
|-------|---------|---------------|
| c_v1 (bake-off) | 0.481 | 45281 real pairs, 80/20 split, all 9 corpus langs |
| c_v2 (plan 114) | 0.481 | 41329 real pairs, 70/30 split, + 8000 synth pairs each for de/es/ru |
| full tier | 120 (fp32 in-repo) | none on this corpus (zero-shot comparator) |

Architecture and hyperparameters are IDENTICAL between v1 and v2
(char-BiGRU 48d/96d/256d, 221k params, symmetric InfoNCE, temp 0.05,
batch 512, 8 epochs, lr 3e-3, occurrence weight cap 8, seed 42): the one
experimental variable is the data. v1 is re-evaluated here as the
no-synth-training ablation; its 80/20 split trains on the perm[70%:80%]
repos the v2 bench holds clean, so v1 rows skip v1-train-tainted pairs
(exclusions counted per row in the JSON reports). The v1 artifact was
regenerated on this branch (provenance note in the manifest receipt).

## Results — top-1 / top-5 / top-20, per component

REAL components (decision evidence — genuinely unseen human typos):

| lang | n | full tier | c_v1 (untainted n) | c_v2 | hybrid | paired net top-5 (95% CI) |
|------|---|-----------|--------------------|------|--------|---------------------------|
| en | 2509 | 0.103 / 0.207 / 0.288 | 0.106 / 0.258 / 0.373 (1616) | **0.128 / 0.283 / 0.406** | 0.119 / 0.270 / 0.406 | +7.5 pp (+5.9, +9.1) |
| de | 14 | 0.214 / 0.429 / 0.714 | 0.500 / 0.625 / 0.750 (8) | **0.429 / 0.643 / 0.786** | 0.286 / **0.714** / 0.786 | +21.4 pp (+0.0, +42.9) |
| ru | 27 | 0.074 / 0.370 / 0.519 | 0.000 / 0.556 / 0.778 (9) | 0.222 / **0.556** / 0.704 | 0.111 / 0.481 / 0.704 | +18.5 pp (-11.1, +44.4) |
| es | 20 | 0.200 / **0.350** / 0.400 | 0.308 / 0.308 / 0.385 (13) | 0.150 / 0.250 / **0.500** | 0.350 / 0.450 / 0.500 | **-10.0 pp** (-30.0, +10.0) |

SYNTH components (generator domain, labeled: v2 trains on this generator,
so C carries a train-distribution advantage by construction; c_v1 does not):

| lang | n | full tier | c_v1 | c_v2 | hybrid |
|------|---|-----------|------|------|--------|
| de | 2000 | 0.113 / 0.280 / 0.378 | 0.296 / 0.609 / 0.795 | 0.328 / **0.699** / **0.893** | 0.214 / 0.544 / 0.893 |
| ru | 2000 | 0.070 / 0.266 / 0.421 | 0.258 / 0.617 / 0.807 | 0.294 / **0.747** / **0.915** | 0.162 / 0.569 / 0.915 |
| es | 2000 | 0.132 / 0.326 / 0.438 | 0.289 / 0.650 / 0.830 | 0.367 / **0.727** / **0.898** | 0.240 / 0.566 / 0.898 |

Latency (batch-1 per suggest, median / p90 ms): c_v2 3.5-4.6 / 4.9-9.3;
hybrid 3.6-5.8 / 4.1-11.3. Size: 0.481 MB int8 (60x inside the 30 MB cap).
Both far inside every cap — size and latency never were the question.

## Decision rule (pre-declared in biencoder-v2.summary.json)

Ship C v2 as an opt-in registry resource only if it (a) beats the full tier
top-5 by >= 5 pp (paired net, CI excluding 0) on en AND de, (b) degrades
nowhere by more than 2 pp top-5 on any language/component with n >= 20 real
pairs, (c) size <= 30 MB int8 and median per-suggest latency <= 100 ms.
An honest reject is a complete outcome (int4 precedent); no registry
change and no tag on a reject.

## Verdict per clause

- (a) **FAIL** (needs BOTH en and de). en: pass — +7.5 pp on 2509 real
  pairs, CI [+5.9, +9.1]. de: the real component (+21.4 pp at n=14) has a
  CI of [+0.0, +42.9] — 0 is inside, so de does not clear the paired-CI bar
  on real evidence; only the labeled synth component clears it (+41.9 pp,
  CI [+39.5, +44.3]), and a generator-domain component cannot carry a ship
  clause.
- (b) **FAIL** — es real, n=20 (>= 20): c_v2 0.250 vs full 0.350 top-5,
  -10.0 pp, beyond the 2 pp allowance. The 20 pairs are statistically thin
  (5 vs 7 hits, sign-test p ~ 0.6) and the rule still binds — that is what
  a pre-declared gate is for. Three independent Spanish real slices now
  agree (bake-off 13 pairs: C 0.308 vs full 0.385; this bench v1 13 pairs;
  this bench v2 20 pairs), so this is not one unlucky draw.
- (c) PASS — 0.481 MB, 3.5-4.6 ms median.

**REJECT.** No registry change, no tag, no release.

## Findings worth keeping

1. The bake-off en result replicates and strengthens at 10x scale: on 2509
   genuinely unseen pairs (vs 257 in the bake-off), c_v2 beats the full
   tier by +7.5 pp top-5 and +11.8 pp top-20 — and v2 training fixed the
   top-1 gap the bake-off flagged (0.128 vs full 0.103; v1 was 0.070 vs
   0.144 on its slice). The signal is real; it is the es real regression
   that fails the ship, not weak English.
2. Generator-domain results do not license a ship. c_v2 dominates the
   noise.py synth components (+40 to +48 pp top-5) while losing the only
   real Spanish slice — training on the generator buys generator wins that
   do not transfer to real Spanish typos. Note c_v1 (never trained on
   synth) already wins synth by +33-38 pp: most of that is architecture
   fit to single-edit noise, not memorization; v2 adds ~+9-11 pp of
   generator-specific gain and simultaneously LOSES ~6 pp of es real
   (0.308 -> 0.250 vs v1). Synth training is not free.
3. The surviving thread — the hybrid (c_v2 top-20 retrieval, fastText full
   rescore): it beats the full tier top-1 AND top-5 on ALL FOUR real
   components (en +6.3 pp, de +28.5 pp, ru +11.1 pp, es +10.0 pp top-5)
   and every synth component, i.e. it is the only measured variant that
   degrades nowhere. It was NOT the pre-declared candidate and is not
   shipped; any future plan must first price its real cost — retrieval
   needs the per-language 100k x 256 C vocabulary matrix (100 MB fp32,
   ~25 MB int8 per language) on top of the 0.481 MB model, plus a real
   Spanish (and German) typo corpus to confirm the es real signal beyond
   20 pairs.
4. What would actually change the verdict: real non-English typo data.
   The corpus contains 212/551/186 unique de/ru/es pairs in total — the
   binding constraint on every de/ru/es conclusion in this plan. A real
   Spanish/German typo corpus (or an expanded extraction) is the
   prerequisite for any retry, not more architecture.

No registry changes, no tier changes, no tags: the fastText tiers remain
the shipped models, exactly as after the bake-off.
