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

## Post-merge live check

`kotoshu setup <lang> --typo` must report `typo: downloaded` (the
plan-137 backfill) and `KOTOSHU_TYPO_RETRIEVAL=1` must arm each
language in well under a second.
