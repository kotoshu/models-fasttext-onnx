# Plan S0: the master graph — dependency order, gates, and the success definition

## Status

authored (2026-09-22). This file is the execution graph for S1-S9: what
depends on what, which gate freezes each node, and what "done" means
for the whole program. Update it whenever a gate verdict lands.

## The graph

```
 WAVE A (data + honesty)          WAVE B (models)              WAVE C (ship + verdict)
 ┌────────────────┐  ┌────────────────┐
 │ S1 typo engine │  │ S5 LLM offline │   (S1 ∥ S5 — independent starts)
 │  layout-gener. │  │  teacher/judge │
 └───┬───────┬────┘  └───┬────────┬───┘
     │splits │pairs      │negatives│distilled pairs
     ▼       ▼           ▼        ▼
 ┌──────────────────┐  ┌──────────────────┐
 │ S8 wave-2 eval   │  │ S2 constrained   │
 │ (freeze baseline │  │ seq2seq (slate-  │
 │  per class×lang) │  │  classification) │
 └────────┬─────────┘  └────────┬─────────┘
          │                     │
          │   ┌─────────────────┘
          ▼   ▼
 ┌────────────────────┐     ┌──────────────────┐
 │ S3 MLM reranker    │     │ S4 CJK confusion │  (feature v1: no model)
 │ (context, rank-    │     │ S9 phonetic      │  (feature: needs S1 accent data)
 │  don't-generate)   │     └────────┬─────────┘
 └────────┬───────────┘              │
          ▼                          ▼
 ┌───────────────────────────────────────┐
 │ S6 KD-QAT int8 + latency lanes        │
 │ (every model ships through here)      │
 └───────────────────┬───────────────────┘
                     ▼
 ┌───────────────────────────────────────┐
 │ S8 re-run → the verdict table         │
 │ (#1 per language × per class or bust) │
 └───────────────────────────────────────┘

 Independent product track: S7 retrieval grounding (no graph deps).
```

## Edge list (why each edge exists)

- S1→S8: splits must come from the same generator that trains the models — eval and train share the error taxonomy, splits stay corpus-free and reproducible.
- S1→S2/S3/S9: unlimited (typo, correction) training pairs; accent-class pairs for S9.
- S5→S2: distilled minimal-edit pairs (edit-level majority-vote filtered).
- S5→S3: hard negatives for the reranker.
- S5→S8: label-noise audit of frozen splits (inter-annotator agreement reported).
- S8→S2/S3/S9: the frozen gate each model must pass (no ship on vibes).
- S2/S3/S9→S6: every artifact compresses through one KD-QAT pipeline; no model ships fp32.
- S4: v1 is a pure ranking feature (confusion-set membership) — no model, no S1 dependency; its model phase consumes S1-class IME pairs and S2's char-level constraint pattern.
- S7: independent; optionally uses S1 to synthesize a project-jargon eval split.

## Gates

| gate | node | pass condition |
|------|------|----------------|
| G-S1 | S1 | generated error-class distribution ≈ human splits (KL report committed); generator reproducible corpus-free |
| G-S8 | S8 | ≥2,000 pairs/language frozen with per-class tags; public repo republish; wave-1 verdicts re-earned |
| G-S2 | S2 | C1 harness per language; out-of-slate output = 0 BY CONSTRUCTION; slate-rerank top-1 lift > 0 on realword |
| G-S3 | S3 | C1 realword slices; FP budget unchanged (rerank-only); ≤10ms/sentence server lane |
| G-S4 | S4 | CSCD-NS + RCSCB through the C1 extension (char-level P/R); confusion-hit feature lifts zh top-1 |
| G-S9 | S9 | accent-class accuracy on wave-2 splits; es/fr realword closes on Hunspell |
| G-S6 | S6 | per-lane latency budgets MEASURED and in manifests; conformance replay green for every tier |
| G-S7 | S7 | personal-domain split lift > flat personal dictionary |

## The success definition (whole graph)

On wave-2 realistic splits: kotoshu ≥ every field lane (SymSpell,
Hunspell, bounded LLM baselines) on every shipped language for nonword
AND realword — realword won via S2+S3+S9+ (S4 for CJK), shipped through
S6's lanes, with every claim reproducible from the public benchmark
repo. Interim success = wave-2 baseline honestly re-earns (or revises)
the current wave-1 claim BEFORE any new model ships.

## Risk fallbacks

- S2 underperforms → the reranking stack (S3+S9) still attacks the class; S2's slate-classification head degrades to a logistic blend.
- S3 misses the browser lane → ship server-lane only first (S6 client-server pattern); mini tier keeps the Ruby engine.
- CSCD-NS/RCSCB licensing friction → substitute the realistic-CSC benchmark set documented in the survey; the gate is the metric shape, not the dataset name.
- S5 cost creep → bounded budgets, outputs cached as committed artifacts, no live API in CI hot loops.

## Consumers

Every S-plan's Status section cites its gate; C8/C6 verdict tables
re-freeze at each S8 re-run.
