# Eval harness for tiered fastText ONNX models

`run_eval.py` measures how well a derived tier model (`fluency`, `mini`)
preserves the ranking behaviour of its full-model parent and gates release
on the measured numbers. Thresholds live in `gates.json`; reports are
written to `eval/reports/{lang}.{tier}.json` and are committed.

## Metrics

### rank_corr (gated)
2000 probe words are sampled across 20 equal rank buckets spanning ranks
50-30000. For each probe, 3000 comparison words are drawn uniformly from the
full vocabulary; comparison words the tier cannot embed (outside the mini
vocab) are dropped from **both** score vectors, so full and tier are scored
on identical word sets. Each model cosine-scores the probe against the
comparison set; the two score vectors are compared with Spearman rank
correlation (`scipy.stats.spearmanr`). The reported value is the mean over
probes.

### top1_agreement (gated, proxy)
400 probe words with rank 1000-30000 each get one deterministic
edit-distance-1 typo (delete / insert / substitute, ascii lowercase). The
typo must itself be in vocabulary (and inside the tier vocab) so both models
can embed it; probes whose typo never lands in vocab after 50 attempts are
skipped and counted. The candidate set is the true word plus the top-20
nearest neighbours of the typo under the **full** model (cosine, over a
20000-word sample window for speed). Candidates outside the tier vocab are
dropped from both models' ballots. Each model scores candidates by cosine to
the typo; agreement is the fraction of probes where both argmax to the same
word.

**Caveat:** this is a proxy for the gem's real spelling-correction rerank
pipeline, not the pipeline itself. The real candidate generator and any
string-prior the gem applies are not modelled here. The report JSON carries
this in `metric_note`.

**Gem conformance slot (kotoshu-rs M3):** when the gem ships conformance
vectors, place them at `eval/conformance/{lang}.json` and score tier outputs
against them; they supersede `top1_agreement` as the release gate. The
harness entry point is `evaluate()` in `run_eval.py` — add a
`conformance` metric alongside the existing ones rather than replacing the
proxy, so historical reports stay comparable.

### coverage (reported, not gated)
Frequency-weighted vocabulary coverage with weight `1/(rank+1)` over the
full vocabulary. The full tier coverage is 1.0 by construction; a 30k-prefix
vocab covers roughly 0.90.

## Determinism
All sampling uses `np.random.default_rng([42, crc32(language)])` (PCG64).
Given the same model pair, reports are reproducible apart from the
`generated_at` timestamp. Both tiers of a language draw the same probe
sequence, so their metrics are comparable.

## Gates (`gates.json`)

| tier    | top1_agreement | rank_corr |
|---------|----------------|-----------|
| fluency | >= 0.95        | >= 0.97   |
| mini    | >= 0.85        | >= 0.90   |

Gates are never weakened. `scripts/build_tiers.py` may only climb its
recipe ladder and records every attempt in the report.

## What the eval decided (2026-09-02, en)

The original recipes cut size with SVD dimensionality reduction
(300→128/144/160 for fluency, →64 for mini). Measured on en, every one
of them failed the fluency gates badly — rank_corr 0.87-0.91 vs the
0.97 gate, top1 0.69-0.76 vs 0.95 — because fastText neighbour
rankings do not survive projection to a low-rank subspace. The recipes
that pass are **full 300 dims + int8 per-row quantization + vocabulary
truncation**: fluency keeps the top 50k words (~15 MB) and mini the
top 10k (~3 MB), scoring rank_corr 0.9999 / top1 1.0000 on the words
they keep. Vocabulary loss is reported separately by the coverage
metric rather than hidden inside the gated metrics — rank_corr and
top1 only score comparison words both models can embed, by
construction (see above).

So: size comes out of the vocabulary and the byte width, never the
dimensions. The eval reports in `eval/reports/` carry the full ladder
history including the failed SVD attempts.

## Usage

```sh
python3 eval/run_eval.py --lang en --tier fluency --repo-root .
python3 scripts/build_tiers.py --lang en de --repo-root .
python3 scripts/build_tiers.py --all --repo-root .
```

Both scripts exit nonzero when a gate fails.
