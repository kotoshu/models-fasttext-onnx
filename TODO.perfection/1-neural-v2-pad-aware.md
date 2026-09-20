# Plan P1: neural scorer v2 — pad-aware training, same frozen gate

## Status

executed with a failed gate, honestly recorded (2026-09-20) — v2 trained the padding exactly as diagnosed (every center, eval-identical windows, 150k steps, loss 0.0143, 99 min) and IMPROVED v1 (true-top at FP 1%: 0.1% -> 1.6%) but failed the gate far below the 60% bar; margins remain 25-48 nats with clean/error overlap, and the canonical probe flipped sign across checkpoints (high variance at this scale). Two tiny-cloze failures establish the rung is exhausted at 27M params / one shard / 150k steps. The next rung - a properly scaled cloze model - is a materially larger owner spend; until then the conservative dual-gate demo is the product ceiling. Evidence: en.probe.neural.json (v2) + en.probe.neural-v1.json.

## Problem

v1 failed the gate for a diagnosed, concrete cause: training used full
16-token windows only, so the PAD embedding is untrained random noise
and short eval contexts (the typo-corpus majority) are
out-of-distribution — margins on a 40-80 nat scale dominated by
padding, not judgment. The signal itself is present and strong (+53.8
nats on real errors). v2 trains the padding.

## What

1. Trainer emits EVERY in-vocab center with a padded window (left
   clipped at sentence start, right at end, PAD-filled) — the exact
   construction the eval scorer uses, so train and eval distributions
   match and the PAD embedding is trained.
2. Same architecture, same data, same steps (150k), same artifact
   contract (fp16, IR 10, checkpoint-before-export).
3. The gate runs on the same frozen split with the same protocol — the
   number is the number. Pass (FP <= 1% AND true-top >= 60%) unblocks
   plans 146/07 and deploys calibrated detection to the demo; fail
   records the next rung honestly.
4. Gate harness batching (the fp16 CPU eval took ~50 min
   single-instance) — only if it does not touch protocol.

## Consumers

gem plan 146, rs plan 07, the demo's detection mode.


## Amendment: v3 (sentence-constructed) — the series' final run

The padding-rate hypothesis was tested to its conclusion: v3 trained
sentences (the eval's own unit, ~90% PAD-heavy centers vs v2's ~2%) and
the gate numbers did not move (true-top at FP 1%: v1 0.1%, v2 1.6%,
v3 1.4% — noise). Three same-scale runs with three different
constructions fail identically: the limit is capacity/data-scale, not
construction. Evidence: en.probe.neural-v{1,2,3}.json. No further GPU
spend on this rung without an explicit owner decision on the scaled
model.
