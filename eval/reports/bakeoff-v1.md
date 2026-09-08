# Plan 111 — embedding bakeoff v1: fastText stays

Question: is a 2017-era context-blind fastText still the right retrieval
model for typo correction, or does a successor candidate in the 15-30 MB
int8 class beat it on the frozen real-typo corpus protocol?

Answer: **fastText stays.** All three candidates are rejected on data.
One (C, the purpose-trained typo bi-encoder) carries the only positive
signal and is the single thread worth a follow-up.

Every number below comes from `eval/bakeoff_bench.py` (pool
construction, ranking rule, tie handling and hit metrics identical to
`eval/corpus_bench.py`, which is untouched) on the frozen vendored
GitHub Typo Corpus, 2000-pair cap, seed 42, `default_rng([42, crc32(lang)])`
sampling. Baseline A numbers are the committed `eval/reports/corpus.{lang}.json`
values, re-verified on this branch (`eval/reports/bakeoff.gates-rerun.json`,
all tier gates pass). Hardware: Apple M1 Max (arm64), CPUExecutionProvider,
onnxruntime 1.23.2. Candidate receipts (sizes, sha256, quantizer, spot-check
parity): `eval/candidates/manifest.json`.

## Decision rule (pre-declared, from bakeoff.summary.json)

A candidate ships only if it beats the fluency-tier corpus top-5 on at
least 3 of 4 languages (en de ru es) AND adds at most 30 MB int8 AND
keeps median per-suggest latency at or under 100 ms. Reject on data is a
complete outcome, same discipline as the int4 ladder.

## Baseline A — current tiers (corpus top-1 / top-5 / top-20)

| lang | pool | full | fluency | mini |
|------|------|------|---------|------|
| en | 2000 | 0.121 / 0.244 / 0.340 | 0.123 / 0.252 / 0.342 | 0.123 / 0.247 / 0.356 |
| de | 72 | 0.167 / 0.417 / 0.542 | 0.143 / 0.397 / 0.540 | 0.119 / 0.333 / 0.524 |
| ru | 139 | 0.180 / 0.410 / 0.583 | 0.169 / 0.444 / 0.589 | 0.160 / 0.467 / 0.587 |
| es | 86 | 0.326 / 0.523 / 0.663 | 0.329 / 0.493 / 0.630 | 0.205 / 0.364 / 0.523 |

## Candidate B — fastText retrieval + ms-marco MiniLM-L-6-v2 int8 rerank

Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, 23.03 MB int8
(inside the size cap). fastText top-100 slate around the typo, reranked
over (context sentence, candidate); a correction outside the slate is a
miss for every k (the product can never surface it).

| lang | fluency top-5 | B top-5 (context) | B top-5 (no context) | slate recall |
|------|---------------|-------------------|----------------------|--------------|
| en | 0.252 | 0.194 | 0.199 | 0.429 |
| de | 0.397 | 0.236 | 0.347 | 0.681 |
| ru | 0.444 | 0.065 | 0.166 | 0.698 |
| es | 0.493 | 0.430 | 0.488 | 0.709 |

**Verdict: REJECT.** Loses to fluency on 4 of 4 languages, and median
per-suggest latency is 805-819 ms (reranking 100 candidates at seq len
96) — 8x over the 100 ms cap. Two structural findings close the class,
not just the checkpoint:

- the retriever is the ceiling: fastText top-100 slate recall is 0.43 on
  English, so no reranker can push en top-5 past 0.43, let alone past
  fluency;
- context from a relevance-tuned cross-encoder HURTS: the no-context
  ablation (query = the typo word alone) beats the context variant on
  all four languages. ms-marco relevance signal is not typo-correction
  signal.

## Candidate C — purpose-trained char-BiGRU typo bi-encoder

`scripts/train_typo_biencoder.py`: shared char encoder (48d chars, 96d
GRU, 256d out), 221k params, **0.481 MB int8** — 60x inside the size
cap. Trained on the vendored corpus, symmetric InfoNCE, temperature
0.05, occurrence-weighted (cap 8), 8 epochs, seed 42, with a STRICT
repo-level split: repos shuffled `default_rng([42])`, 15977 train /
3995 held-out, no repo on both sides, and eval pairs additionally
required to occur only in held-out repos (pair seen in any train repo
is tainted and excluded). Best probe hit@5 0.806 at epoch 6.

Because the corpus is dominated by English (143956 en edits vs 824 ru,
270 de, 254 es), the honest comparison is against the fastText FULL
tier restricted to the identical clean subset (same 100k-word ranking
universe C uses):

| lang | clean n (excluded) | C top-5 / top-20 | full tier on same subset top-5 / top-20 |
|------|--------------------|------------------|------------------------------------------|
| en | 257 (1743) | **0.214** / **0.342** | 0.144 / 0.233 |
| de | 8 (64) | **0.625** / **0.625** | 0.375 / 0.625 |
| ru | 9 (130) | 0.444 / **0.778** | 0.444 / 0.667 |
| es | 13 (73) | 0.308 / 0.385 | 0.385 / 0.462 |

**Verdict: REJECT as a drop-in successor (0-1 of 4 vs fluency on the
full pools), PROMISING as the follow-up direction.** It is the only
candidate that beats fastText on genuinely unseen pairs — +7.0 pp top-5
and +10.9 pp top-20 on the 257-pair English clean subset — at 0.5 MB
and the lowest measured latency (3.7-5.2 ms median, below even fastText
query+matvec at 6.9-9.1 ms). What it needs before a ship decision:

1. non-English training signal (per-lang edits in the receipt: the
   corpus is 98% English, and es/de/ru clean subsets of 8-13 pairs are
   not decision-grade);
2. a larger repo-clean eval split (the 80/20 repo split leaves only
   257/2000 English pool pairs evaluable — an explicit held-out-repos
   bench corpus is the fix);
3. a top-1 story (C top-1 on clean en is 0.070 vs fastText 0.144 — the
   bi-encoder recovers the correction in the top-5/top-20 but rarely
   ranks it first; a hybrid C-retrieve + fastText-rescore is the obvious
   first experiment).

## Candidate D — ModernBERT-base int8 zero-shot bi-encoder

`answerdotai/ModernBERT-base`, mean-pooled L2-normalized, 149.91 MB
int8 — 5x OVER the size cap (ModernBERT-small does not exist on the hub;
recorded `not_found_on_hub` in the receipt, same discipline as the
model2vec potion-mini slot). Measured anyway to price the class.

| lang | fluency top-5 | D top-5 | D top-20 | vocab UNK frac (first 20k) |
|------|---------------|---------|----------|----------------------------|
| en | 0.252 | 0.040 | 0.063 | 0.000 |
| de | 0.397 | 0.111 | 0.194 | 0.000 |
| ru | 0.444 | 0.237 | 0.482 | 0.000 |
| es | 0.493 | 0.035 | 0.105 | 0.000 |

**Verdict: REJECT.** Zero-shot subword mean-pooling does not model
edit-distance structure: top-5 collapses to 0.03-0.24 with a 0.000 UNK
fraction — this is a representation failure, not a vocabulary failure.
Also 5x over size and 36-42 ms median per suggest (5-6x fastText). A
purpose-trained encoder of this class would have to relearn what C
learns for 0.5 MB.

## Latency (median / p90 ms per suggest, batch 1)

| candidate | en | de | ru | es |
|-----------|----|----|----|----|
| A full fastText | 7.3 / 9.6 | 7.9 / 9.8 | 9.1 / 17.3 | 6.9 / 10.8 |
| B rerank slate-100 | 819 / 903 | 809 / 905 | 805 / 890 | 807 / 894 |
| C bi-encoder query+matvec | 5.2 / 8.9 | 3.7 / 11.1 | 4.5 / 12.1 | 4.6 / 11.9 |
| D ModernBERT query+matvec | 41.8 / 74.2 | 38.8 / 68.0 | 37.8 / 70.0 | 35.6 / 53.9 |

C and D per-language vocabulary embedding matrices are amortized (built
once; ~10 min per language for D on this hardware, cached under
`eval/candidates/cache/`, gitignored).

## Verdicts

| candidate | int8 MB | decision |
|-----------|---------|----------|
| B cross-encoder rerank | 23.03 | REJECT — loses 4/4, 8x latency cap, context ablation negative |
| C typo bi-encoder | 0.481 | REJECT as drop-in; PROMISING — needs non-en training data, bigger clean bench, top-1 story |
| D ModernBERT-base | 149.91 | REJECT — representation failure, 5x size cap |
| D ModernBERT-small | — | unavailable (not published) |

No registry change, no tier change, no release: the fastText tiers
remain the shipped models. Rejection on data closes the int4-style
ladder for B and D; C is the open thread.

## Reproduce

```sh
python3 scripts/prepare_bakeoff_candidates.py --repo-root .   # B + D weights (outside git)
python3 scripts/train_typo_biencoder.py --repo-root .         # C training + receipt
python3 eval/bakeoff_bench.py --repo-root .                   # reports to eval/reports/bakeoff.*
```
