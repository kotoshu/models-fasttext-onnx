# Plan 17: the neural context scorer (real-word detection, next rung)

## Status: proposed — blocked on the owner's decision to fund the training arc

## Problem

Plans 15/16 exhausted the non-neural ladder with evidence: the
ranking capability exists (the true correction wins argmax on 65.5%
of covered instances) but every n-gram margin family — summed and
conjunctive, MLE and discounted — produces IDENTICAL margin
distributions on clean and error text (tau=0 FP equals true-top in
all four variants). Real-word detection needs a scorer whose
probabilities are calibrated for context, not a better threshold on
the same margins.

## What (sketch; the funded plan details it)

- One small neural LM per language (or one multilingual model —
  first owner fork in the road) trained on the same Wikipedia shards
  already licensed and fetched (plan 16 infrastructure): score(t |
  left, right) over a +-10 token window, subword/char tokenization
  so any vocab word is scoreable, ONNX-exportable, int8, target
  20-30 MB per language to fit the LFS/browser budget.
- Local training first (the bake-off trained char-BiGRUs locally in
  ~12 CPU-minutes; a windowed scorer is bigger but laptop-scale);
  the Modal recipe exists if it outgrows CPU.
- Zero error-labeled training data needed for the scorer itself (LM
  objective); calibration and the gate reuse EXACTLY the plan-15/16
  harness: same frozen real-word split, same clean sentences, same
  gate (eligible-token FP <= 1% AND covered true-top >= 60%). No new
  benchmark, no re-cut eval.
- On pass: unblock gem plan 146 + rs plan 07 (status flips), package
  the ctx artifact to the design doc's ONNX shape, registry resource
  type follows in their PRs.

## Evidence the rung can clear the bar

The 65.5% argmax capability is the n-gram's CEILING, achieved with
profoundly miscalibrated probabilities; a calibrated context model
converts capability into separated margins. The four-variant
distributional failure (design doc Phase-1 section) is the
credential that no cheaper fix remains.

## Consumers

gem plan 146 and rs plan 07 (engine halves, currently blocked);
docs/realword-detection-design.md gains the Phase-2 section when
trained.
