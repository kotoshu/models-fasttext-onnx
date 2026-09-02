# 01 — Publish All Models & Reconcile README

## Goal

The repo's `README.adoc` claims **6 models, 686MB**. Local clone has
**158 `.onnx` files, 35GB**. Pick the actual v1 set, publish them, and
make the docs match reality.

## Why

`ALL_157_LANGUAGES.md` exists in the repo, suggesting a bulk conversion
was attempted for all of FastText's supported languages. The README
still says 6. The Kotoshu gem's `ModelCache` will fetch from this repo —
if the README is wrong, the gem's language matrix is wrong, and users
discover the discrepancy by surprise.

## Tasks

1. **Inventory every model.** `scripts/inventory.rb` walks `models/`,
   verifies each `.onnx` loads (via a quick onnxruntime check), records
   language code, dimension, vocabulary size, file size, SHA-256.
2. **Decide v1 published set.** Three options:
   - **(a) All 158.** Maximum language coverage; large repo.
   - **(b) Top 30 by speaker count.** Practical balance; matches the
     gem's `Kotoshu.supported_languages` v1 target.
   - **(c) Just the 6 currently advertised.** Smallest; out of step
     with the local state.
   Recommendation: **(b)** for v1.0.0, expand to (a) by v1.1.
3. **Reconcile README.** Update model count, total size, language list,
   and the per-language table. Remove the inconsistency between
   `README.adoc` (6) and `ALL_157_LANGUAGES.md` (157+).
4. **Quarantine unpublishable models.** Any `.onnx` that fails to load
   or has zero vocabulary moves to `models-unverified/` for later
   triage. Don't delete (project rule).
5. **Per-language directory layout** standardized:
   ```
   models/{code}/
     fasttext.{code}.onnx
     vocabulary.txt        # word list, one per line
     metadata.json         # dims, vocab size, source, conversion date
   ```
6. **`models/index.json`** at the repo root cataloging every published
   model with its metadata. The gem reads this to know what's
   available without enumerating dirs.

7. **Storage strategy for scale.** If the published set grows beyond
   GitHub LFS's 100 GB soft limit, pick one: GitHub Packages for
   releases; split into regional repos (`-eu`, `-asia`, `-mideast`);
   or external blob storage (S3, Azure Blob). Document the chosen path
   in this plan's Status before crossing 50 GB.

## Acceptance criteria

- README model count matches `find models/ -name '*.onnx' | wc -l`
- `models/index.json` exists and is the gem's source of truth
- Every published model has a `metadata.json` with non-zero vocab size
- No model in `models/` fails to load

## Dependencies

- Blocks: `kotoshu/TODO.impl/05-semantic-path.md` (gem needs to know
  which models are real), `kotoshu/TODO.impl/03-dynamic-download.md`
- Coordinate with `dictionaries/TODO.impl/02-coverage-matrix.md`
  (model coverage ⊆ language coverage)

## Status

**Partially superseded — v1 active set shipped as 9 languages** (de en
es fr ja ko pt ru zh; commit 0eec64a "trimmed active ONNX catalog"), a
variant of option (b) chosen for v1: one catalog the gem actually
resolves, tiered ×3 by plan 06, distributed via the registry + releases
(plan 07). The 158-model local corpus stays on disk untouched — never
deleted; expanding the active set is now a registry operation (add
manifest entries + tiers), id scheme supports all 157.

Delivered: `manifest.json` (9 × onnx + 9 × vocab, sha256/size),
per-language `metadata.json` + `vocab.json` (the vocabulary.txt sidecar
idea shipped as vocab.json), `registry.json` replaces the `index.json`
idea. README: registry/distribution section lands with plan 07; the
model-count reconciliation pass over README.adoc/README.md remains
open, as does the load-verification inventory sweep (`inventory.rb`)
for anything newly added.

## Source

Reference data: `docs/ALL_157_LANGUAGES.md` (full FastText 157-language
inventory from Grave et al. 2018), `docs/SETUP_REPORT.md` (baseline
verification of the 6 already-converted models — de/en/es/fr/pt/ru,
each 100K vocab × 300D × 114.44 MB, IR v11 opset 11, load < 1 s,
inference ~1–2 ms/query, ~115 MB resident). Both are historical
snapshots — verify against current `models/` before relying on any
number.
