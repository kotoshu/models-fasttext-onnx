# 1 — Final verdict freeze

## State

The final wave (16 languages, gem b7c5106a + gated splits) is landing
on Modal: 14/16 reports in the volume; ru and pt remain (their first
freezes ever), plus the ten re-freeze overwrites (en de es it ko nl
zh-Hans-CN zh-Hant-TW enqueued in the second spawn; pl fr ja vi
zh-Hant-HK ar already landed final).

## Steps

1. Volume reaches 16/16 → `modal run scripts/bench_modal.py::collect`.
2. Freshness audit per file: a final-era report is identifiable by its
   FIELD lane numbers (the gated splits changed both lanes — e.g. pl
   field 88.0→86.35, vi 65.15→66.4, ar 61.5→65.8). Any language whose
   report still carries pre-gate field numbers was not overwritten —
   re-spawn it before declaring.
3. Freeze `eval/reports/` (committed) + build the verdict table:
   per language — kotoshu t1/t3/t5 vs best field lane, nonword AND
   realword, the ≥-gate result, and the generation lineage.
4. Update TODO.sota-impl/2 + the master graph with the final numbers.
5. Update TODO.sota-impl/11 status from benches-freezing to frozen.

## Gate

Every number in the table traces to a committed report on the final
engine; the website announcement consumes nothing else (TODO.final/5).
