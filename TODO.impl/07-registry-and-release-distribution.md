# 07 — Registry & Release Distribution (Resource Spec v1)

Executes `02-storage-strategy.md` (hybrid: text in git, binaries as
release assets) and `05-versioned-releases.md` (tags + per-tag
manifest), extended with the tier dimension from
`06-tiered-models.md` and the registry shape defined in
`kotoshu/TODO.impl/65-universal-kotoshu.md` (Resource Spec v1).

## Goal

One canonical, versioned registry the Ruby gem, kotoshu-rs, the
server, and the SDKs all resolve against — with binaries on GitHub
Releases (unmetered on public repos) instead of growing Git LFS.

## The registry

`registry.json` at the repo root (the existing `manifest.json` remains
as a generated compatibility view for the gem's current resolver until
the gem's tier-aware client lands — then it is deprecated, not
deleted). Shape (Resource Spec v1):

```json
{
  "spec": "kotoshu.resources/v1",
  "registry_version": 2,
  "generated_at": "…",
  "release_tag": "v1.0.0",
  "resources": {
    "kotoshu://models/en/fluency": {
      "type": "model",
      "language": "en",
      "tier": { "name": "fluency", "dims": 128, "vocab_size": 100000,
                 "quantization": "int8-per-row" },
      "version": "1.0.0",
      "urls": {
        "primary": "…/releases/download/v1.0.0/fasttext.en.fluency.onnx",
        "mirror":  "…/raw/…/models/en/fasttext.en.fluency.onnx"
      },
      "vocab_url": "…/releases/download/v1.0.0/fasttext.en.fluency.vocab.json",
      "sha256": "…", "size_bytes": 13000000,
      "license": "CC-BY-SA-3.0",
      "min_engine_version": "0.7",
      "eval_ref": "eval/reports/en.fluency.json"
    }
  }
}
```

Rules:

- `id` is stable forever; content changes bump `version` and the
  release tag, never the id.
- `urls.mirror` points at the git tree (LFS today) — a fallback if a
  release asset is unavailable; primary is always the release asset.
- `min_engine_version` is advisory metadata for consumers; the floor
  itself is an owner decision recorded here, not chosen by tooling.

## Distribution layout (executes plan 02's Option C)

- **In git**: `registry.json`, per-tag `manifest-v{TAG}.json`,
  `models/**/metadata.json`, vocab files, `eval/reports/**`, scripts,
  docs. Existing LFS-tracked full models stay exactly where they are —
  they remain the archival source and the mirror; nothing is deleted
  (global rule). New tiers are **not** pushed to LFS.
- **Release assets** (per tag): every `.onnx` of every tier +
  per-tier vocab. Asset names match the file names one-to-one.
- **Fresh clone without LFS stays < 100 MB** (plan 02 acceptance),
  minus the pre-existing pointer-only blobs.

## Release workflow (executes plan 05)

GitHub Action `release.yml`, triggered by tag push (tag pushed only by
the owner — never automated):

1. Validate `registry.json` against the JSON Schema
   (`schemas/registry.schema.json`, added by this plan): ids unique,
   sha256/size spot-checked against the local files, tier blocks
   present, eval gates referenced and passing.
2. Create the GitHub Release; upload every referenced asset.
3. Write `manifest-v{TAG}.json` into the tag (release asset) and
   commit the registry bump to `main` via PR.
4. Release notes generated from the registry: languages × tiers,
   sizes, upstream sync dates, ONNX opset, eval summary.

### URL contract (v1 API, locked)

```
https://github.com/kotoshu/models-fasttext-onnx/releases/download/{TAG}/{asset}
https://raw.githubusercontent.com/kotoshu/models-fasttext-onnx/{TAG-or-main}/registry.json
```

The gem's existing base-URL override and pin mechanism
(`KOTOSHU_REPOS_BASE_URL` and the per-repo pins) extends to pin
`{TAG}`; unpinned consumers track the registry's `release_tag`.

## Bandwidth & fallback

Release assets on public repos are unmetered; LFS was the quota risk
and stops growing under this plan. Document a monthly bandwidth
estimate and a mirror fallback (per plan 02) in `docs/storage-decision.md`.

## License & attribution

Upstream `cc.{lang}.300.vec` vectors are CC BY-SA 3.0; derived tiers
inherit it. Every registry entry carries `license`; README gains an
attribution section; the site's language pages link it. **Attribution
wording is an owner decision** — flag before the first release.

## Tasks

1. `schemas/registry.schema.json` + `scripts/build_registry.py`
   (manifest + release tag → registry.json).
2. `scripts/upload_release_assets.py` (gh CLI) + `.github/workflows/release.yml`.
3. First release cut — **tag name and version numbers are the owner's
   decision**; the workflow refuses to invent them.
4. `manifest.json` compatibility view generation + deprecation note.
5. `docs/storage-decision.md` (decision record + bandwidth estimate).
6. README: registry documentation, URL contract, attribution.

## Acceptance

- `registry.json` validates; every resource resolves to a real release
  asset with matching sha256.
- A fresh clone without LFS is < 100 MB.
- The gem (current resolver, via `manifest.json`) keeps working
  unchanged before the tier-aware client lands.
- Release notes are generated, not hand-written.

## Dependencies

- Blocked by: `06-tiered-models.md` (tiers must pass gates before
  they are registered).
- Blocks: gem plan 67 M1/M2 (tier-aware ModelCache, release pin),
  kotoshu-rs P3, `05-versioned-releases.md` close-out.

## Status

**Implemented (2026-09-02)** — `schemas/registry.schema.json` (draft
2020-12), `scripts/build_registry.py` / `validate_registry.py` /
`release_notes.py`, `.github/workflows/release.yml` (tag-gated,
eval-gated, never creates tags), `docs/storage-decision.md`, README
registry + URL-contract section, and a dev-flavor `registry.json` at
the repo root: **27 resources, 9 languages × 3 tiers, validated with
full local hash checks, 0 warnings**. First release pending the
owner's tag — the workflow refuses to invent one.
