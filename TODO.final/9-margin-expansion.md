# 9 — Margin expansion: widening the winning gap across all languages

Current margins (16/16 WIN, avg +1.6pp): the tight ones are ko +0.2,
zh-Hant-TW +0.3, zh-Hans-CN +0.7, ru +0.9, de +1.0, en +1.1. The
it/ko probes showed the residual losses are distance-1 tie lotteries —
frequency decides, and frequency is a coin the field flips too.

## The levers, by evidence and priority

1. **S2 slate-reranker** (TODO.sota-impl/3): the fasttext context
   model as a suggest-tier tiebreaker — when distance and rank tie,
   the model that detects realword errors picks the intended word.
   The models already exist for all 57 languages; this is THE
   margin-expander (turns coin-flips into wins wherever the context
   model has an opinion). Gate: frozen splits re-freeze, kotoshu
   margin grows, no nonword regression.

2. **Distance-3 expansion**: our both-miss rates are 25-35% (vi 31.7%,
   ar ~30%) and many are distance-3 corrections the field (symspellpy
   TOP, max-2) cannot reach at all. Our engine can afford distance-3
   (the deletion index already supports it; max_dist is a constant).
   Gate: nonword top-1 gain measured against distance-3 precision
   (the false-positive rate must not eat the gain); the field lane
   stays at max-2 — that asymmetry IS the engine difference.

3. **Fold-scoring audit on de/sv**: the vi lesson (fold scoring
   inverted tone ranking) generalized to pl; de/sv are still
   fold-sanctioned from pre-clean-split evidence. Test de with raw
   scoring on the gated splits — if the margin grows, the sanctioned
   set shrinks to nothing and the code simplifies.

4. **Kelly rank refinement from real corpora**: top-1 accuracy is the
   RELATIVE order of the target among its distance-1 neighbors. The
   kelly blends carry approximate rank in the tail; a per-language
   head re-ordering from real corpora directly improves tie outcomes.

5. **Layout-adjacency as a Latin tie-break signal**: generalize the
   CJK confusion-hit bonus — when distance and frequency tie, prefer
   the candidate whose substitution is keyboard-adjacent. This is how
   humans actually typo (the split classes prove it: adjacent-sub is
   the top class everywhere).

## The honesty guard

Every lever uses an INDEPENDENT signal (context model, layout
physics, real-corpus frequency, deeper search) — not split-derived
patterns. The splits are frozen; engine changes that grow the margin
on independent evidence are legitimate; anything fitted to the class
tags is overfitting the test and is forbidden.

## Non-levers (already maxed)

The capability column (cross-script) and the realword column are
structurally ours — no field lane can enter them. The remaining
realword gap is the S3 context arc (sentence tier).
