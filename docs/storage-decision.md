# Storage decision: hybrid git + GitHub Releases

Status: decided. Implements Option C of `TODO.impl/02-storage-strategy.md`
via `TODO.impl/07-registry-and-release-distribution.md`. This document is
the decision record and the v1 URL contract.

## Decision

Small text lives in git; every `.onnx` binary lives as a GitHub Release
asset. Release assets on public repositories are unmetered, which removes
the quota pressure that made Git LFS the bottleneck.

| Location | Contents |
|---|---|
| git (plain) | `registry.json`, `manifest.json`, per-tag `manifest-v{TAG}.json` copies, `models/**/metadata.json`, `models/**/tiers.json`, vocab files (`*.vocab.json`), `eval/reports/**`, scripts, schemas, docs |
| GitHub Releases (per tag) | every `models/**/fasttext.*.onnx` (all tiers), every `models/**/fasttext.*.vocab.json`, `registry.json`, `manifest-v{TAG}.json` |
| Git LFS (frozen) | the existing full models, exactly as they are today |

## Git LFS

- The existing LFS-tracked full models are the archival source and the
  mirror. They are never deleted and never rewritten.
- `.gitattributes` stays untouched so the existing blobs keep resolving.
- New tier artifacts (`fasttext.{lang}.{mini,fluency}.onnx` and their
  vocabs) never enter LFS and are never committed: they are build
  outputs of `scripts/build_tiers.py`, gitignored, and published only as
  release assets. The LFS footprint therefore stops growing at its
  current size.
- **Addendum (2026-09-05, plan 77):** the "frozen" wording above
  described the tier-quantization work. Plan 77's coverage expansion
  (9 → 22 languages) deliberately grows LFS again: each promoted
  language's full `.onnx` (~120 MB) enters LFS exactly as the original
  nine did, because the registry's full-tier mirror URL is the media
  host and the whole promotion flow (`.gitignore` negation block →
  `generate_manifest.rb` → `build_tiers.py`) is built around the full
  model living in the git tree. Tier artifacts remain LFS-free. Public
  repositories get LFS storage/bandwidth free of charge, so the
  previous quota pressure does not return; footprint after plan 77 is
  ~2.6 GB (22 full models) instead of ~1.1 GB.

- **Addendum (2026-09-05, plan 83):** batch 2 grows coverage from 22 to
  55 languages — every dictionaries-repo language fastText Common Crawl
  vectors can serve. Same promotion flow and rationale as plan 77: each
  promoted language's full `.onnx` (~120 MB) enters LFS, tier artifacts
  stay LFS-free. Footprint is now ~6.5 GB (55 full models, ~3.9 GB of
  them added by this batch). The registry full-tier mirror continues to
  point at the media host.

## Bandwidth

- Primary downloads hit release assets: unmetered on public repos.
- `raw.githubusercontent.com` serves only small text (registry,
  manifests, vocabs, metadata) — negligible traffic, well under the
  soft cap of roughly 10x repo size per month.
- LFS bandwidth is frozen at current usage and keeps shrinking as a
  share of total traffic.
- Fallback order for consumers: release asset (primary) → mirror URL in
  the registry (`/raw/main/...`, meaningful for full models and vocab)
  → retry. The mirror is fallback-only, never the default. Note that
  tier artifacts have no git mirror until the LFS question is revisited;
  their only durable copy is the release asset.
- If raw traffic ever grows beyond comfort, a CDN mirror in front of the
  release assets is the planned escape hatch (plan 02 fallback).

## v1 URL contract (locked)

```
https://github.com/kotoshu/models-fasttext-onnx/releases/download/{TAG}/{asset}
https://raw.githubusercontent.com/kotoshu/models-fasttext-onnx/{TAG-or-main}/registry.json
```

`{asset}` is the file's basename, one-to-one with the repo layout
(`fasttext.{lang}[.{tier}].onnx`, `fasttext.{lang}[.{tier}].vocab.json`).
This pattern is the v1 API; changes require a registry spec bump, not
silent edits. Consumers pin `{TAG}` (the gem's `KOTOSHU_REPOS_BASE_URL`
override and per-repo pins extend to it); unpinned consumers track
`release_tag` in `registry.json`.

## Release mechanics

`.github/workflows/release.yml` runs only on a pushed `v*` tag — the tag
is always cut manually by the owner. It builds the tiers (eval gates are
the release gate), builds and validates the registry, snapshots
`manifest-v{TAG}.json`, generates the notes from the registry, and
uploads every asset.

## Fresh-clone budget

A clone without LFS pulls only the text layer (vocabs ~17 MB total,
registry, manifests) — comfortably under the 100 MB target of plan 02.
The 120 MB full models arrive via release assets or an explicit
`git lfs pull`.
