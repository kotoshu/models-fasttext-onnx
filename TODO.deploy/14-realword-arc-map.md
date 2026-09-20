# Plan D14: the real-word arc's map is complete — one owner-gated rung remains

## Status

executed (2026-09-20, ninth issuance of the standing directive)

## Problem

The directive's demo sentence drives at real-word detection, and the
ninth pass re-read the design record end to end. The finding: the arc's
deploy-reachable surface was already complete, and the one thing that
remains is not engineering anyone can start — it is an owner decision
recorded in the design doc itself.

## What

1. **The frozen verdict, restated for the record**: the Phase-1 bigram
   context gate RAN on English and failed distributionally — four
   margin variants (sum/conjunctive × MLE/discount), all showing clean
   and error margins as the same distribution; at the 1% FP budget,
   recall collapses to ~5% while argmax ranking (65.5% true-top) proves
   the signal exists but no n-gram threshold separates it. Trigram
   shares the failure mode. Evidence: `eval/realword/en.probe.ctxlm.json`.
2. **The demo's copy now states this honestly** (updated in this plan):
   it flags the high-confidence collocational slab and cites the frozen
   verdict — no implied promise of calibrated detection.
3. **What the deploy arc contributed to the detection goal, all green**:
   en + de context tables (94M / 102M tokens), confusion tables frozen
   for en/de/es/fr/pt/ru, table-v2 recipe (62.2% coverage), turnkey
   corpus fetching, the daily serving watch, and the 57-language
   browser proof of everything that ships today.

## The decision (the arc's funnel point)

Everything below the next rung is done. The next rung is **plan 17:
the neural context scorer** — a small masked-LM trained per language
over word-in-context, a genuinely different model class. It unblocks,
upon passing the same gate: gem plan 146, rs plan 07, and per-language
detection in this demo. It costs a training arc (Modal is proven:
~$0.10/run for the substrate; the neural arc is larger — measured, not
assumed, before scaling). Per the design record and the standing rules,
that spend is the owner's call.

## Consumers

The owner's decision; plans 146/07 and the demo's detection mode wait
on it.
