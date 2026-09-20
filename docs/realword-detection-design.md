# Real-word (confusion-set) detection — design record

Status: Phase 0 complete (evidence below). Engine plans (gem 146, rs 07)
remain blocked until the Phase-1 context model clears the gate.

## Why this exists

Detection everywhere is vocabulary membership (gem
`semantic_analyzer.rb` `next if valid_word?`; rs check path the same):
an in-vocab word is never questioned, so real-word errors — "I want to
**each** rice", 再/在 — are invisible, and every model we ship only
ranks corrections for words already flagged as non-words. As the owner
put it: a checker whose detection is a dictionary is a dictionary with
good autocomplete attached. The models must participate in the
detection decision, not only the suggestion ranking.

## Mechanism

For each in-vocab token with a non-empty confusion set:

```
flag(word, ctx)  ⇔  max_{c ∈ confusions(word)} [ S(c, ctx) − S(word, ctx) ] > τ_lang
```

- `confusions(word)` — precomputed in-vocab mis-selection candidates
  (per-language table artifact).
- `S(c, ctx)` — a context scorer. The product contract is the FP
  budget: `τ_lang` frozen from calibration, never tuned in an engine.
- Errors carry the argmax candidate as the suggestion; a distinct
  error kind so clients render them differently from misspellings.
- Opt-in at the API surface until calibration earns default-on.

## Phase 0 evidence (en, GitHub Typo Corpus real-word split)

Eval base: `eval/realword/en.json` — 12,648 unique pairs / 54,073
edits where BOTH sides are in the 100k full-tier vocab (the class a
dictionary gate cannot see), with sentence contexts;
`en.clean.jsonl` 105,700 corrected sentences for the FP side.
Confusion table v1 (`eval/confusion/en.json`, DL≤1): 314,641 pairs,
mean degree 9.2, 10 MB.

Two hard ceilings found:

**1. Table coverage.** DL≤1 covers only **15.6%** of real-word
instances (3,209/20,517 with context). The dominant classes are
distance-2: you→your, occured→occurred. No scorer can exceed this
through the v1 table.

**2. Scorer separation.** Operating points anchored on clean-text
margins (flag-rate over scoreable error instances; true-top = flagged
with the TRUE correction as argmax, over covered instances):

| Scorer | FP 0.5% | FP 1% | FP 2% | FP 5% |
|---|---|---|---|---|
| cosine margin (the shipped CosineReranker math, ±5 tokens) | 4.3% | 8.1% | 13.4% | 25.1% |
| frequency prior (vocab rank gap; context-free baseline) | 11.5% | 16.1% | 28.9% | 44.8% |

The static-embedding context scorer is WORSE than raw frequency —
cosine-to-neighbors does not encode the syntactic/collocational fit
the decision needs. Clean-text margins and error margins overlap
almost completely.

**3. The skipgram rung is unavailable upstream.** The ladder's next
step was fastText's own conditional scorer S(w|ctx) = out(w)·Σ in(c)
from the training binaries. Verified at true offsets
(`scripts/extract_output_matrix.py`; input 4,000,000×300, qout=0,
output 2,000,000×300): the shipped cc.*.300 output matrices are
**all zeros** — the distributed .bin files carry no trained output
matrix. The conditional scorer must be TRAINED, not extracted.

Reproduce: `scripts/extract_realword_pairs.py --lang en`,
`scripts/build_confusion_tables.py --lang en`, and
`scripts/eval_realword_detection.py --lang en --scorer {cosine,freq}`
(frozen evidence: `eval/realword/en.probe.{cosine,freq}.json`); the
zeroed-output finding via `scripts/extract_output_matrix.py`.

## Phase 1: the context model (the only remaining rung)

Train a compact word n-gram LM per language — S(w | w₋₂, w₋₁, w₊₁, w₊₂)
with stupid-backoff, over a web-scale corpus (Wikipedia per language):

- **Why n-gram, not a transformer**: collocational fit ("want to
  each/eat rice", "harder then/than") is exactly what bigram/trigram
  tables capture; the artifact is int8-quantizable, ~10–40 MB per
  language, loadable in rs/wasm/gem without a runtime dependency, and
  its margins are inspectable (debuggability a transformer denies us).
  A tiny transformer stays a possible Phase 5 if n-gram misses the
  gate — measured, not assumed.
- **Artifact**: `fasttext.{lang}.ctx.onnx` — tensors {ngram_ids int64,
  q_scores int8, row_scale f32} keyed by hashed n-grams, same
  int8-per-row convention as buckets; metadata `model_type=ctx_lm`,
  `orders: [2, 3]`, `bo: 0.4`. Consumed by the rs `confuse` module and
  the gem through the existing provider seam.
- **Training data decision is the owner's** (Wikipedia licensing is
  fine; crawl dumps are heavy). en first, gate before scaling.

**Gate to unblock plans 146/07** (numbers are a proposal for the owner
to bless): eligible-token FP ≤ 1% AND covered-instance true-top
recall ≥ 60% on this same eval split, with the table-v2 coverage.

## Phase 1 evidence (en, bigram ctx-LM — GATE FAILED, ladder escalated)

Table v2 (the Phase-0 "coverage" half of the gate): with the
identity-variant deletion-index fix (v1 silently missed EVERY
insert/delete pair between len>=2 words — you/your and
occured/occurred are distance 1, not 2; the Phase-0 "distance-2
dominates" reading was partly that bug) plus bounded DL<=2
(one side in the top-30k band) plus a lowercase-alpha gate (cased
vocab entries can never be observed), coverage rises
**15.6% -> 62.2%** at 1,031,283 pairs / mean degree 54. The table is
a keeper regardless of scorer outcome.

The scorer half: a bigram LM over one Wikipedia shard
(wikimedia/wikipedia 20231101.en, 94.3M in-vocab tokens, 10.5M
bigram types, stupid backoff a=0.4, additive unigram smoothing,
min-support 100 on context words, adjacent-neighbor context only).
Three scorer bugs were found and fixed on the way, each recorded
because they change how to read any future rerun: conditioning on the
target instead of the context word; inverted margin polarity; and
zero-count candidates scoring a vacuous 0.0 "perfect fit".

Verdict (frozen in eval/realword/en.probe.ctxlm.json):

| Point | flag (errors) | true-top (covered) | FP (clean) |
|---|---|---|---|
| tau=0 (argmax) | 86.8% | **65.5%** | 70.5% |
| FP-anchored 1% | 8.3% | 5.1% | 1.0% |

The ranking capability is there — the true correction wins argmax
65.5% of covered instances, ABOVE the gate's 60% bar — but the
margin distributions on clean and error text overlap almost
completely: at the operating point where the FP budget (1%) is met,
recall collapses to 5.1%. A candidate-frequency proximity filter
(1x-1000x sweeps) and the min-support gate do not separate them.
This is structural for n-gram scoring: max-over-~50-candidates
margins are positive for ~70% of CORRECT words (some plausible
neighbor always edges out the true word under sparse MLE), the same
magnitude the error class produces.

**Separation exhausted (post-verdict experiments, same frozen
split):** two further margin families were tested before accepting
the FAIL — conjunctive aggregation (the candidate must beat the
observed word on BOTH adjacent bigrams, not their sum) and absolute
discount smoothing (d=0.75) — four variants total:

| Variant | tau=0 true-top | tau=0 FP | @FP 1% true-top |
|---|---|---|---|
| sum / MLE | 66% | 69% | 5.8% |
| conj / MLE | 55% | 55% | 4.8% |
| sum / disc | 66% | 69% | 5.6% |
| conj / disc | 55% | 57% | 4.4% |

At tau=0 the error and clean margin distributions are the same
distribution (FP ≈ true-top in every variant) — no monotone
threshold on these margins can separate them. The failure is
distributional, not a calibration artifact.

**Ladder conclusion (trigram shares the failure mode — same
sparse-MLE noise class): the next viable rung is a neural context
scorer (small masked-LM over word-in-context), a genuinely different
model class and training arc — plan 17, an owner decision. The gate
stands unweakened; gem plan 146 and rs plan 07 remain blocked.**

Reproduce: `scripts/build_confusion_tables.py --lang en --band 30000`,
`scripts/train_ctx_lm.py --lang en --corpus <shard> --source ...`,
`scripts/eval_realword_detection.py --lang en --scorer ctxlm`.

## Confusion table v2

- Add bounded distance-2 (the dominant real-word class): d=2 only
  where at least one side is in the top-N frequency band (N ≈ 20–30k),
  via 2-deletion indexing; expected table size to be measured before
  adoption (v1 is 10 MB at d≤1).
- Add phonetic-class pairs (sound-alike: their/there) per language.
- CJK phase: homophone tables (pinyin readings for zh, kana for ja)
  over the tier vocab — a table SOURCE, no engine change; the sparse
  seed pairs already extracted (zh 22, ja 23, ko 20) grow with a
  CJK typo corpus in a later phase.

## FP budget and rollout

Phase 0 (done): evidence above. → Phase 1: train + gate the en ctx-LM,
build table v2. → Phase 2: gem plan 146 (analyzer branch, opt-in API).
→ Phase 3: rs plan 07 (native scoring, wasm/server). → Phase 4: CJK
homophone tables. → Phase 5: registry resource type + cut, default-on
decision, remaining languages.

The FP budget is a product contract: wrong flags are the worst UX a
checker delivers, so real-word detection ships opt-in and earns
default-on only through the gate above.


## Phase 1 evidence (en, neural context scorer v1 — GATE FAILED, cause diagnosed as trainable, not distributional)

Plan 17 authorized (owner, 2026-09-20). A 27.3M-parameter word-level
cloze transformer (d=256, 2 layers, tied embeddings, gap-position
encoding over the same 100k vocabulary, +/-8 window) trained 150k steps
on the en wiki shard (final loss 0.014; Modal A10G, 78 min; artifact
fasttext.en.ctx-neural.onnx, 52 MB fp16, IR 10).

Verdict (frozen in eval/realword/en.probe.neural.json):

| Point | flag (errors) | true-top (covered) | FP (clean) |
|---|---|---|---|
| FP-anchored 1% | 3.7% | 0.1% | 1.0% |
| FP-anchored 5% | 12.2% | 0.8% | 5.0% |

GATE FAILED — but the failure mode differs fundamentally from the
n-gram verdict. The margins live on a 40-80 NAT scale dominated by
padding artifacts: training used FULL 16-token windows only, so the PAD
embedding is untrained random noise, and every short eval context
(sentence edges — the majority of typo-corpus sentences) is
out-of-distribution. Probes confirm both halves: real errors score
+53.8 (eat over each in "want to each rice") while CLEAN short-window
text also scores +33.8 — the signal is present and strong; the padding
noise swamps it at the operating points. Unlike the n-gram's identical
clean/error distributions, this is a CONCRETE, fixable defect:

  v2 = pad-aware training (emit padded windows at sentence edges so the
  PAD embedding is trained, or add an attention mask input), same gate.

The ladder stands unweakened; plans 146/07 remain blocked pending v2.
Reproduce: scripts/modal_train_ctx_neural.py (see --steps),
scripts/eval_realword_detection.py --lang en --scorer neural.


## Phase 1 evidence (en, neural v2 pad-aware — GATE FAILED; the tiny-cloze rung is exhausted at this scale)

v2 (TODO.perfection/1) trained the padding exactly as diagnosed: every
in-vocab center, eval-identical padded windows, same architecture/data/
steps (150k, loss 0.0143, 99 min). Verdict frozen in
eval/realword/en.probe.neural.json (v1 preserved as
en.probe.neural-v1.json):

| Point | flag (errors) | true-top (covered) | FP (clean) |
|---|---|---|---|
| FP-anchored 1% | 6.5% | 1.6% | 1.0% |
| FP-anchored 10% | 28.5% | 9.6% | 10.0% |

GATE FAILED — improved over v1 (true-top 0.1% -> 1.6% at FP 1%) but
nowhere near the 60% bar, and the margin scale remains 25-48 nats with
clean/error overlap. Two consecutive tiny-cloze failures with frozen
protocol establish: at 27M params / one wiki shard / 150k steps this
model class does not separate. The canonical probe even flipped
checkpoints (eat +53.8 in v1, eat -6.5 in v2) - high variance across
runs at this scale.

Ladder status: cosine, frequency, bigram (4 variants), tiny-cloze v1
and v2 have all failed the same frozen gate. The next rung is a
PROPERLY SCALED cloze model (10-100x parameters or training scale) -
a materially larger owner spend, or the conservative dual-gate demo
remains the product ceiling. The gate stands unweakened; plans 146/07
remain blocked.


## Phase 1 evidence (en, neural v3 sentence-constructed — GATE FAILED; three-run series complete)

v3 (TODO.perfection, third issuance) split training into sentences so
sentence-edge PAD-heavy centers dominate exactly as in eval (~90% vs
v2's ~2% paragraph-boundary rate). 150k steps, loss 0.0141, 98 min.
Verdict frozen in eval/realword/en.probe.neural-v3.json (v1/v2
preserved alongside):

| Point | flag (errors) | true-top (covered) | FP (clean) |
|---|---|---|---|
| FP-anchored 1% | 5.8% | 1.4% | 1.0% |
| FP-anchored 10% | 28.7% | 9.9% | 10.0% |

GATE FAILED — statistically indistinguishable from v2 (1.6% at FP 1%).
The padding hypothesis is DISPROVEN as the dominant factor: training
the padding at the eval's own rate moved nothing. Three consecutive
same-scale runs (v1 untrained-PAD, v2 paragraph-rate PAD, v3
sentence-rate PAD) land in the same place, with clean/error margins
overlapping on a 25-50 nat scale in every variant. The series
establishes: at 27M parameters over one Wikipedia shard, this model
class cannot separate - the failure is capacity/data-scale, not
construction. THE LADDER TERMINATES at an owner decision: a properly
scaled cloze model (10-100x parameters or multi-shard training) at
materially higher spend, or the conservative dual-gate demo as the
product ceiling. The gate stands unweakened; plans 146/07 remain
blocked.
