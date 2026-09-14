# Plan 13: bulk KTM1 matrix builder

## Status: executed

## Problem

Plans 136 and 12 derived matrices by hand (one language per cargo
invocation, hand-written descriptors). The next language to gain a
matrix would re-learn the same sequence. The rebuild of English through
the bulk path is byte-identical to the plan-136 artifact
(`sha256 df1a7b2c…`), so the derivation is deterministic and the
orchestration is the only missing piece.

## What

- `scripts/build_typo_matrices.py` — bulk orchestration around the
  kotoshu-rs `matrix_export` example:
  - `--langs de,it,…` (comma-separated)
  - `--typo-onnx` / `--typo-vocab` (the frozen bi-encoder pair)
  - `--tiers-dir` optional (else `models/{lang}/fasttext.{lang}.onnx`)
  - vocab sidecars always from this repo (`models/{lang}/…vocab.json`)
  - writes `typo.matrix.{lang}.ktm1` + `typo-matrix.json` (descriptor
    shape matches plans 136/12, including the paired full-tier sha from
    the live registry)
  - `--install` copies into `models/{lang}/`
  - refuses to proceed without the KTM1 magic / exporter rc=0

## Evidence

- Rebuild of `en` through the script: 26 000 016 bytes,
  `sha256 df1a7b2cd55015ef…` — byte-identical to the plan-136 artifact
  (27 s wall).
- `--help` renders; missing-tier paths fail with a clear error.

## Next language recipe

```bash
# 1. fetch the full-tier onnx (sha-verify against registry)
# 2. cargo build -p kotoshu --features model --release --example matrix_export
python3 scripts/build_typo_matrices.py \
    --langs <code> \
    --typo-onnx ~/.cache/kotoshu/models/typo/typo.biencoder.onnx \
    --typo-vocab ~/.cache/kotoshu/models/typo/typo.biencoder.vocab.json \
    --tiers-dir /path/to/tiers --install
python3 scripts/build_registry.py --tag v1.7.0
python3 scripts/validate_registry.py --registry registry.json --check-files
# PR; CI --check-urls (plan 11) proves the branch media host serves it
```
