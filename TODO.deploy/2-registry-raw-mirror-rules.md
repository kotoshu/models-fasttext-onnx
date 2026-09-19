# Plan D2: registry raw-mirror rules — CORS surfaces without LFS

## Status

executed (2026-09-19) — 232 resources, 0 errors/0 warnings; mini/packs/lid/typo/en-buckets carry raw-host mirrors, 169 release-only entries have null mirrors, ktm1 primaries at v1.8.0

## Problem

The registry's mirror convention points at the LFS media host. After plan
D1 there are no LFS objects, so every mirror URL must either move to the
raw host or become null. The rule must be statable in one sentence:

> Mini tiers (and the browser resources: packs, LID, typo, and bucket
> tables of demo languages) carry a raw-host mirror; everything else is
> release-only.

## What

1. `scripts/build_registry.py`: `MEDIA_URL` → `RAW_URL`
   (`https://raw.githubusercontent.com/kotoshu/models-fasttext-onnx`);
   mirrors for `mini` tiers, packs, lid, typo, and
   `BROWSER_BUCKET_LANGS = {"en"}` buckets point at
   `{RAW_URL}/{ref}/<path>`; `full`, `fluency`, non-demo `buckets` get
   `mirror: null` (release-only); typo-matrix (ktm1) entries flip from
   "mirror-only until a release carries it" to release primaries at the
   registry tag (the v1.8.0 release carries the six matrices).
2. `scripts/validate_registry.py`: mirror the same expectations; the
   typo-matrix branch's primary-null assertion becomes a primary-at-tag
   assertion; released fluency/buckets/full must have null mirrors.
3. Full-tier vocab handling unchanged (full vocabs stay plain git, and
   their `vocab_url` stays the release asset). Fluency vocab urls stay
   release assets.
4. Rebuild with `--tag v1.8.0` (no `--unreleased`), structural
   validation green, zero mirror-only full tiers, zh still retired.
5. Registry consumers: the gem is already null-mirror-safe
   (`[primary, mirror].compact.uniq`); py/rs/js read primary-first.

## Consumers

- Plan D3 validates URLs against the live refs after the cut lands.
- Plan D4's demo fetches `registry.json` and the en mini/vocab/buckets
  mirrors this rule produces.
