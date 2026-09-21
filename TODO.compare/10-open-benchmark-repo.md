# Plan C10: the open benchmark repo — validate us, configure your own lanes

## Status

shipped (2026-09-21) — kotoshu/suggesters-benchmark (public, MIT):
extract_pairs.py (GitHub Typo Corpus → labelled splits; corpus never
redistributed), benchmark.py (the C1 harness with lanes.yaml
discoverable engine lanes and --only filtering), frozen RESULTS in
README (en #1 on every slice; de top-3/5/realword #1), environment
disclosure (Ruby 3.4, PR-#226 ranking, published frequency lists).

## Problem

"We are #1" is only credible if outsiders can re-run the comparison
and if the lanes are configurable instead of hardcoded.

## What

1. lanes.yaml as the configuration surface: engine lanes + dataset
   lanes + scoring config; harness discovers lanes from the file.
2. Reproduction instructions matching our environment exactly.
3. Future: CI lane that re-runs the en benchmark on gem releases so
   the numbers on the README can never silently rot (needs a runner
   with the compiled gem — follow-up).

## Consumers

Anyone validating our claims; every future contender we benchmark;
C8's fleet loop (same harness, more dataset lanes).
