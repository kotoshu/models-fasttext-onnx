# Impl S2: constrained slate reranker (graph node S2)

## Status

PIPELINE SHIPPED; gate NOT yet passed (2026-09-22) — three honest
results: (1) char cross-encoder from scratch, 3-8 epochs: 63-75% vs
82.5% sym baseline — FAIL; (2) rank-embedding variant, 30 epochs:
71.4% — worse (overfit); (3) 6-feature logistic blend on HUMAN
SymSpell-only slates: +1.8pp WIN (88.6 vs 86.8). KEY FINDINGS: human
truths sit in the SymSpell slate only 48% of the time but in the full
composite slate 96.1% — the rerank target is the COMPOSITE slate;
synthetic data cannot demonstrate reranker value (inverse-generator
problem, see impl 1). NEXT (defined, not yet run): neural reranker
with a PRETRAINED backbone trained on the 10k human en pairs +
composite slates (dump script shipped: scripts/dump_slates.rb);
logistic blend is the S0-fallback shipping candidate. Shape decision: the constrained decoder IS a slate classifier — a compact cross-encoder scoring (typo, candidate_i) pairs, K=10 slate; free generation impossible by construction. Deliverables: Modal training script (multilingual MiniLM-class backbone or small BERT, ~30-80M); train on S1 pairs (+S5 distilled when budgeted); ONNX IR-10 export; Ruby integration (TypoSlate strategy/rerank over the SymSpell slate); C1 gate.

## Gate

G-S2: out-of-slate = 0 by construction; realword top-1 lift > 0 on frozen splits; per-language reports.

## Consumers

Real-word class; es/fr vs Hunspell; kotoshu-rs parity roadmap.
