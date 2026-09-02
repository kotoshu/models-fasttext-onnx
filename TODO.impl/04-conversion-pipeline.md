# 04 — Conversion Pipeline & CI

## Goal

A reproducible Python pipeline that converts upstream FastText `.vec`
files to ONNX, with CI jobs that detect upstream changes and reconvert.

## Why

The repo currently has `scripts/` (175 files) including conversion
scripts, but the process isn't reproducible — there's no version-pinned
environment, no manifest of which upstream `.vec` files were used, no
CI to detect when FastText upstream releases new vectors. A future
maintainer can't regenerate the models from scratch with confidence.

## Tasks

1. **Conversion entry point.** One canonical script,
   `scripts/convert_one.py --language en --vec-url ... --out
   models/en/`. Wraps the existing `convert_fasttext_to_onnx.py`
   logic.
2. **Upstream pin file.** `scripts/upstream_versions.json` recording,
   per language:
   - FastText file URL
   - FastText file SHA-256
   - Release date
   - Conversion tool version (the Python deps)
3. **Reproducible environment.** `scripts/requirements.txt` with
   pinned versions; `scripts/setup_python.sh` installs into a venv.
   Document Python 3.10+ requirement.
4. **Batch conversion.** `scripts/convert_all.py` iterates
   `upstream_versions.json` and runs the conversion. Idempotent:
   skips languages whose output already matches.
5. **CI job.** Weekly action:
   - Polls FastText's CDN for new `.vec` files
   - If any differ from `upstream_versions.json`, opens a tracking
     issue
   - Doesn't auto-reconvert (a human reviews first)
6. **Conversion test.** A spec/test that converts a tiny toy model
   (10 words, 5-dim) and verifies the ONNX loads and produces
   expected output. Catches conversion regressions.
7. **Vocabulary sidecar.** Each conversion emits
   `models/{code}/vocabulary.txt` so the gem can load words without
   reading the ONNX.
8. **Provenance in `metadata.json`.** Every model records:
   - upstream `.vec` URL + SHA-256
   - conversion script version
   - onnxruntime version it was validated against
   - exact conversion command

## Acceptance criteria

- `scripts/convert_one.py --language en` produces a model identical
  (byte-for-byte) to the committed one
- `upstream_versions.json` records every source
- CI weekly job reports upstream changes
- A new maintainer can reconverte the full set in <24 hours

## Dependencies

- Blocks: `01-publish-all-models.md` (need reproducible conversion
  before declaring v1 set)
- Coordinate with `02-storage-strategy.md` (conversion output feeds
  the storage layout)

## Status

**Partially implemented.** Shipped: `scripts/fasttext_to_onnx.py` as
the canonical converter (deterministic graph build, opset 11,
metadata_props provenance) and — via plan 06 — `scripts/build_tiers.py`
extending the pipeline with the mini/fluency tier derivation + eval
gates, run in CI at release time.

Still open: `upstream_versions.json` pin file (upstream `.vec` URLs +
SHA-256 + dates), pinned `requirements.txt`/venv setup, the weekly
upstream-poll CI job (opens a tracking issue, never auto-reconverts),
the toy-model conversion regression test, and byte-for-byte
reproducibility verification of the committed full models.
