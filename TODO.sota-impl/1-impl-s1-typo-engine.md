# Impl S1: synthetic typo engine (graph node S1)

## Status

EXECUTED (2026-09-22). Generator shipped (scripts/generate_typos.py,
38 layout grids exported from the gem via scripts/export-layouts.rb);
7 error classes incl. IME pinyin-confusion (CC-CEDICT homophone
groups) and diacritic-omit; wave-2 splits frozen for 12 languages
(2,000 nonword each; realword 200 for the big six, 38-40 for it/nl/pl,
0 for CJK — honestly out of scope until S4's confusion layer, which
IS the CJK realword class). En-route fixes: unicodedata NFD case law,
bounded realword sampler (dense-vowel languages thrash unbounded).
FINDING: the synthetic distribution is the generator's own inverse —
SymSpell is near-optimal on it (86.8% top-1) and NO reranker can
demonstrate value on it; human splits are the only honest training/
eval source for S2/S3 (48.2% truth-in-SymSpell-slate vs 96.1% in the
full composite slate). Deliverables: (a) `scripts/export_layouts.rb` — layout adjacency grids exported from the gem registry to JSON; (b) `scripts/generate_typos.py` — per-language generator: adjacent/non-adjacent substitution (layout-distance-weighted), transposition, double-letter insert/delete, diacritic omission (fold-equal, per C9), pinyin-confusion (zh, from CC-CEDICT), realword neighbor-swap (valid word at edit distance 1); (c) wave-2 splits per language with per-pair class tags; (d) realism report — edit-distance distribution + class mix vs human splits.

## Gate

G-S1: KL(edit-distance dist || human dist) committed; generator runs corpus-free.

## Consumers

S8 splits; S2/S3 training pairs; S9 accent pairs.
