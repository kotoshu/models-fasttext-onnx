# Plan C2: the scaled cloze context model — both axes at once

## Status

in-progress (2026-09-21, owner-authorized spend)

## Problem

The three-run series (v1/v2/v3, TODO.perfection/1) proved 27M
parameters over one wiki shard cannot separate clean from error
margins; the design record names capacity/data-scale as the limit.
The owner authorized the scaled run ("do all of these properly").

## What

Scale both axes in one run, keeping the frozen gate unchanged:

- capacity: d=384, 4 layers, 6 heads, FFN 1536 → ~46M params
  (fp16 artifact ~92 MB — under the 100 MiB git limit)
- data: 3 Wikipedia shards (en train-00000..00002) instead of 1
- training: 150k steps, batch 256, same window/construction as v3
  (sentence-split, eval-identical padding), Modal A10G, checkpoint
  before export (the law)

Gate: scripts/eval_realword_detection.py --lang en --scorer neural on
the same frozen split; FP ≤ 1% AND true-top ≥ 60% unblocks plans
146/07. A fourth failure at 46M/3-shard scale escalates the ladder
conclusion to "this recipe needs a materially different scale or
class" — recorded either way.

## Consumers

The real-word detection arc; plans 146 (gem) and 07 (rs) remain
blocked until this gate passes.
