# Plan 22: full-tier ONNX binaries leave git-LFS history — release-only storage

## Status

executed (2026-09-18) — 55 full-tier `.onnx` paths stripped from `main` and
`fleet-retrains` history via `git filter-repo`; both branches force-pushed
(owner authorized "A! DO IT NOW!" 2026-09-17, "OK do it!" 2026-09-18).
Pre-flight gate: all 55 files verified byte-identical (sha256) to their
v1.7.0 release assets BEFORE any history rewrite. Builder + validator carry
the release-only mirror rule. Evidence in `eval/reports/` and the push
outputs recorded in this file.

## Problem

GitHub LFS storage is at ~90% (the org data pack). The killer is the
full-tier tier ladder step: 55 languages × ~120 MB = **6.60 GB** of
`models/{lang}/fasttext.{lang}.onnx` LFS objects at tip, plus every
historical version of those paths. Meanwhile every full-tier binary
already exists, byte-identical, as a **v1.7.0 release asset** — the
canonical copy moved to releases when the fleet landed release-only
(commit 09302b5). The LFS copies are pure duplication, and the quota
block they cause is what stranded the fleet metadata push.

The browser does NOT load the full tier: the tier ladder serves browsers
`mini` (~3 MB) / `fluency` (~15-18 MB) / `buckets` (~11-22 MB) through the
LFS media host (the only GitHub surface with `Access-Control-Allow-Origin:
*`). Full tier is a server/CLI artifact, fetched via the registry
`primary` URL from release downloads. So removing full-tier binaries from
git costs the browser nothing.

## What

1. **Safety gate**: verify all 55 full-tier files are byte-identical to
   v1.7.0 release assets (sha256 vs the release `digest` field).
   Result: 55/55 match, 0 mismatch. THE precondition for the strip.
2. **Preserve round-2 fleet models**: the second ca/hu/uk/vi retrain round
   (Sep 18 02:28) sat uncommitted; its mini/fluency onnx + vocab metadata
   match the release assets (digests match the working tree). Backed up to
   `~/src/kotoshu/models-backup-20260918/`, mini/fluency tiers committed
   on `fleet-retrains`; full-tier round-2 binaries stay release-only
   (backup + release are the copies).
3. **Registry rule (builder + validator)**: a released full-tier entry
   (primary non-null at the registry tag) is **release-only** —
   `urls.mirror` is null, because no LFS object exists to serve. Every
   other tier (mini/fluency/buckets) and every pre-release language
   (mirror-only, e.g. the zh variants until the owner's cut) keeps the
   media-host mirror. The schema already allowed null mirrors ("null when
   the artifact exists only as a release asset") — only the builder's
   unconditional media URL and the validator's hard expectation change.
4. **History strip**: `git filter-repo --partial --refs main
   fleet-retrains --invert-paths --paths-from-file <55 paths>`. Tip LFS
   drops from ~9 GB to ~2.5 GB (small tiers 1.69 GB + zh variants 0.45 GB
   + int4 experiments 0.33 GB + packs/ktm1/lid/typo); historical versions
   of the stripped paths leave git too. `--partial` keeps the local LFS
   cache and old objects (no local gc) — nothing is destroyed locally.
5. **Force-push** both branches (owner-authorized; history rewrites cannot
   go through PRs). Tags are NOT pushed (owner law); see Consumers for
   the residue that leaves.

## Consumers

- **gem / kotoshu-py / wasm**: unchanged flow — `primary` release URLs
  serve full tier; small tiers keep media mirrors for browser CORS.
- **Registry consumers reading `mirror`**: a null mirror on a released
  full tier now means "release-only"; the schema documented this
  semantics before this plan made it real.
- **Server-side LFS reclaim**: old commits stay reachable through the 10
  release tags (`v1.0.1`…`v1.7.0`) and merged-PR refs, so GitHub's
  storage number only fully drops after the owner re-cuts tags and/or
  GitHub support garbage-collects unreferenced LFS objects. The push
  block lifts regardless once no new objects exceed the pack.
- **Next registry cut (v1.8.0, owner's call)**: ca/hu/uk/vi + zh variants
  promote from mirror-only to release primaries; full tiers stay
  release-only forever after.
