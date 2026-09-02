# 05 — Versioned Releases

## Goal

Tag `v1.0.0` and ship a release with a stable URL contract for the gem
to pin against.

## Why

Same reasoning as the other content repos: the gem needs version
pinning. Additionally, because this repo is the largest (multi-GB),
release tags double as the storage vehicle (per `02-storage-strategy.md`
option B/C) — the gem resolves `{code}/fasttext.{code}.onnx` to a
release asset URL.

## Tasks

1. **First release: `v1.0.0`.** Cuts after `01-publish-all-models`,
   `02-storage-strategy`, `03-manifest-checksums`,
   `04-conversion-pipeline` land.
2. **Semver policy:**
   - ONNX IR version bump or breaking format = major
   - New languages = minor
   - Re-converted from updated upstream = patch
3. **Per-tag manifest** at `manifest-v{TAG}.json`.
4. **Release assets.** Each `.onnx` uploaded as a release asset on the
   tag. Asset URL pattern documented in the README.
5. **Release notes** per tag listing:
   - Languages added
   - Upstream FastText sync dates
   - ONNX IR version
   - Total size
6. **Release automation.** GitHub Action on tag push:
   - Validates manifest
   - Uploads each model as a release asset (or marks the LFS pointer
     if option A)
   - Cuts the Release with notes
7. **Gem URL contract.** The Kotoshu gem's `ModelCache` resolves:
   `https://github.com/kotoshu/models-fasttext-onnx/releases/download/v{TAG}/fasttext.{code}.onnx`
   Lock this URL pattern as a v1 API.

## Acceptance criteria

- `v1.0.0` tag exists with every v1-published model as a release asset
- `manifest-v1.0.0.json` is reachable at the tag
- The gem's default `model_sources` URL points at `v1.0.0`, not `main`
- Asset URLs match the documented pattern

## Dependencies

- Blocked by: `01-publish-all-models.md`, `02-storage-strategy.md`,
  `03-manifest-checksums.md`, `04-conversion-pipeline.md`
- Blocks: `kotoshu/TODO.impl/11-release-v1.md`

## Status

**Implemented via [07-registry-and-release-distribution.md](07-registry-and-release-distribution.md)** — `.github/workflows/release.yml`
is tag-gated (tags pushed only by the owner, never automated): builds
tiers, enforces eval gates, builds + validates the registry, generates
release notes from it, uploads every `.onnx` + vocab + `registry.json`
+ `manifest-v{TAG}.json` as assets. URL contract locked:
`releases/download/{TAG}/{asset}`. The semver policy rows above stand.
**First release pending the owner's tag** — the version number is the
owner's decision; the workflow refuses to invent one.
