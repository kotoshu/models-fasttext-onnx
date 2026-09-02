# 03 — Manifest & Checksums

## Goal

Repo-level `manifest.json` enumerating every published `.onnx` model
with its size, SHA-256, dimension, and vocabulary size — same format
as the other two content repos.

## Why

The Kotoshu gem's `ModelCache` currently downloads `.onnx` files
without any integrity check. A 230MB+ binary fetched over HTTPS with
no verification is the largest attack surface in the project. Cross-
references `kotoshu/TODO.impl/09-integrity-security.md`.

## Tasks

1. **Manifest schema** (identical to the other content repos):
   ```json
   {
     "version": 1,
     "generated_at": "...",
     "resources": {
       "models/en/fasttext.en.onnx": {
         "size": 230000000,
         "sha256": "...",
         "language": "en",
         "type": "onnx",
         "dimension": 300,
         "vocabulary_size": 100000,
         "fasttext_source": "cc.en.300.vec",
         "fasttext_version": "...",
         "conversion_date": "..."
       }
     }
   }
   ```
2. **Generate.** `scripts/generate_manifest.rb` walks `models/`,
   loads each model briefly to extract dimension and vocabulary size,
   hashes the file, emits `manifest.json`.
3. **CI check.** PRs must update the manifest; CI diffs and fails on
   drift.
4. **Sidecar metadata.** Each `models/{code}/metadata.json` is the
   per-model detail; the root `manifest.json` is the aggregate. The
   gem uses the manifest; tools and humans use the sidecars.
5. **Per-tag snapshot.** `manifest-v{TAG}.json` per release tag.
6. **Signature (stretch).** Sign the manifest with minisign/sigstore.

## Acceptance criteria

- `manifest.json` exists at repo root with entries for every
  published model
- `scripts/generate_manifest.rb` is idempotent and runs in <5 minutes
  (the bottleneck is loading each model briefly)
- CI blocks PRs with stale manifests
- Schema matches `dictionaries/TODO.impl/01-manifest-checksums.md` and
  `frequency-list-kelly/TODO.impl/02-data-validation.md`

## Dependencies

- Blocks: `kotoshu/TODO.impl/09-integrity-security.md`,
  `kotoshu/TODO.impl/05-semantic-path.md`
- Should land in lockstep with the other two repos' manifest plans

## Status

**Closed (2026-09-02).** Core shipped earlier (manifest.json with
size/sha256 per resource); strengthened by plan 07 (schema validation,
release-time hash re-verification, per-tag manifest snapshots). The
PR-time drift check now exists: `.github/workflows/ci.yml` regenerates
the manifest and fails on any diff beyond `generated_at` (regeneration
verified byte-stable locally). Remaining stretch, deliberately open:
minisign/sigstore signing — requires owner-held keys, so it is an
owner-gated future item, not an engineering gap.
