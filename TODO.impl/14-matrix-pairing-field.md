# Plan 14: matrix↔tier pairing travels the registry and is enforced

## Status: executed

## Problem

A KTM1 matrix's rows are index-parallel to EXACTLY the full-tier
vocabulary they were derived over. That pairing existed only in the
descriptor on disk — the registry entry carried nothing, and nothing
anywhere enforced it. If a full tier were ever rebuilt (new vocab
order, new sha), the shipped and cached matrices would pair with the
wrong vocabulary and arm silently-wrong slates: every row would
retrieve a different word than it was quantized for.

## What

- Registry entries for `typo-matrix` resources gain
  `paired_vocab_sha256` (parsed from the descriptor's `paired_vocab`
  by the generator; nullable in the schema, hex-pattern-checked).
- The validator enforces it twice:
  - construction: the field must be present and a 64-hex sha;
  - ground truth: the entry's pairing must equal the full tier's
    manifest sha — a rebuilt tier without a rebuilt matrix fails with
    "rebuild the matrix (scripts/build_typo_matrices.py)".
- registry.json regenerated at the v1.7.0 tag: the diff is the six
  pairing shas (de/ec1100…, en/cf8267…, es, fr, pt, ru — each matching
  its language's live full tier) plus `generated_at`.
- Tests: the field rides generation and validates green; a mismatched
  pairing fails with the rebuild recipe; a deleted field fails
  construction.

## Consumers

The gem's `ModelRegistry` (lutaml-model) tolerates unknown fields
(proven by parse test), so this lands additively. Plan 142 (gem) reads
the field to reject a stale pairing at arm time.
