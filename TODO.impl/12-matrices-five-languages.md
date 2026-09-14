# Plan 12: prebuilt KTM1 matrices for de, es, fr, pt, ru

## Status: executed

## Problem

Plan 136 shipped the prebuilt typo-retrieval matrix for English only.
The other five full-feature languages (de, es, fr, pt, ru) derive the
index at first arming (~25 s) or never get instant arming at all. The
hybrid's measured wins are largest on exactly these languages (de
+28.6 / ru +11.1 / es +10.0 top-5 points vs the full tier), so the
slowest arming was paid by the best customers.

## What

- `models/{de,es,fr,pt,ru}/typo.matrix.{lang}.ktm1` — KTM1 v1,
  100 000 rows × 256 dims, 26 000 016 bytes each, derived with
  `kotoshu-rs examples/matrix_export` from the sha-verified full tiers
  (downloaded from the v1.7.0 mirrors, registry shas matched) and the
  language's vocab sidecar from this repo.
- `models/{lang}/typo-matrix.json` descriptors (same shape as en's:
  mirror-only until a release carries the artifact; rows pair with the
  full-tier vocab, sha-recorded).
- registry.json regenerated at `--tag v1.7.0` (main's committed state
  is the tagged generation): the diff is exactly the five new entries
  plus `generated_at`. 223 resources.
- Committed through LFS via the `*.ktm1` rule (PR #40).

## Validation

- `validate_registry.py --check-files` — OK, 223 resources.
- `--check-urls --urls-ref <branch>` (plan 11's ref-aware probe) — the
  CI gate proves the five mirrors serve from the branch; plan 10's
  pre-merge gate could not have done this.

## Post-merge live check (2026-09-14, PR #43 merged)

- `kotoshu setup de --typo` on a wiped registry: `typo: downloaded`,
  matrix cached at 26 000 016 bytes, 24.5 s total for spelling + pair +
  full tier + matrix.
- de arming: **511 ms** via the matrix vs **44 917 ms** derived (88x);
  both engines answer identically on probe words (artifact==derived
  parity holds, the property the rs round-trip test freezes).
- es/fr/pt/ru matrices fetched live through `download_typo_matrix`:
  KTM1 magic, 26 000 016 bytes, sha-verified, 2.2-6.6 s each.
