# 02 — Storage Strategy

## Goal

Decide how the multi-gigabyte model set is stored: plain git, Git LFS,
or GitHub Release assets. Implement the decision.

## Why

The local clone is 35GB for 158 models. A full `git clone` of that size
is hostile to contributors, CI runners, and the gem itself (which
shouldn't clone the repo — it should fetch individual files via raw
URLs). GitHub soft-recommends LFS for files >50MB; release assets for
files >2GB.

## Tasks

1. **Decision document.** `docs/storage-decision.md` laying out:
   - Option A: Git LFS. Pros: transparent to git tooling. Cons: bandwidth
     quotas, LFS-as-asset model.
   - Option B: GitHub Releases. Pros: no LFS quota; release assets can be
     any size. Cons: not browsable in the tree; need a side manifest.
   - Option C: Hybrid. Plain git for `metadata.json`, `vocabulary.txt`,
     manifest; LFS or releases for `.onnx` files.
   Recommendation: **Hybrid (C)** — small text in git, large `.onnx` in
   GitHub Releases attached to language-specific tags.
2. **Migration script.** Whatever option is chosen, `scripts/migrate.rb`
   moves existing files into the new layout without losing any
   `.onnx` (per the global rule — never delete).
3. **URL contract for the gem.** Whatever layout we pick, the gem's
   `ModelCache` resolves `{code}/fasttext.{code}.onnx` to a stable URL.
   Document the URL pattern and lock it as a v1 API.
4. **Bandwidth budget.** Estimate monthly GitHub bandwidth for typical
   Kotoshu usage; if it exceeds GitHub's soft caps (typically 10x the
   repo size per month), plan for a CDN mirror.
5. **LFS tracking file** (if option A or C with LFS): `.gitattributes`
   with `models/**/*.onnx filter=lfs diff=lfs merge=lfs -text`.
6. **Release asset upload** (if option B or C with releases): the
   release-automation action uploads each `.onnx` as a release asset
   on language-specific tags.

## Acceptance criteria

- A fresh `git clone` (without LFS) is under 100MB
- The gem can download any model without cloning the repo
- Storage decision is documented and locked for v1
- Bandwidth estimate is documented with a fallback plan

## Dependencies

- Blocks: `01-publish-all-models.md` (need to know the layout before
  publishing), `kotoshu/TODO.impl/05-semantic-path.md`
- Coordinate with `kotoshu/TODO.impl/09-integrity-security.md` —
  whatever storage we pick, checksums must be verifiable

## Status

**Implemented via [07-registry-and-release-distribution.md](07-registry-and-release-distribution.md)** — Option C exactly:
small text (registry, manifest, vocabs, metadata, eval reports) in git;
every `.onnx` (full + tiers) as GitHub Release assets (unmetered on a
public repo); existing LFS full models frozen in place as the archival
mirror, nothing deleted; new tiers gitignored and never enter LFS.
Decision record: `docs/storage-decision.md`. Fresh-clone-without-LFS
target re-anchored on the trimmed 9-language catalog. CDN-mirror
fallback documented as a future option if raw traffic ever demands it.
