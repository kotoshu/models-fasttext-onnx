# Plan C2: scaled cloze model — the honest scale test

## Status

executed and FAILED the frozen gate (2026-09-21). Scaled run: 45.6M params (d_model=384, layers=4, heads=6, ffn=1536), 3 en shards, sentence construction, A10G. Training hit a Modal ConnectionError (Deadline exceeded) around step 56k/150k; the lander fetched whatever was exported and ran the gate. Result at FP ≤ 1%: true-top **1.4%** (need ≥ 60%). At FP 10%: true-top 9.9%. Evidence: /tmp/c2-gate.log + eval/reports/en.probe.neural-v* lineage. Scale did not open the gate. The real-word class remains context-bound and unsolved by a cloze transformer of this size on this corpus. Plans that depended on a passing gate (146 neural consumer, 07 browser real-word beyond n-gram) stay blocked on a different approach — not another scale bump of the same design.

## Problem

v1/v2/v3 tiny-cloze all failed the same frozen gate (0.1%/1.6%/1.4% at FP 1%). The hypothesis was that scale (params × data) would open it.

## What

Train a larger cloze transformer on Modal A10G, export ONNX IR-10, run the frozen gate unchanged. Record the verdict either way.

## Consumers

The real-word product surface; any plan that assumed a neural context scorer.
