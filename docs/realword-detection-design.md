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
