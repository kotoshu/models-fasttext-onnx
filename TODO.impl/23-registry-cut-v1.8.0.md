# Plan 23: registry cut v1.8.0 — fleet + zh variants promote, `zh` retires

## Status

executed (2026-09-18, owner authorized "consider it unblocked" for the
v1.8.0 cut) — registry rebuilt at tag v1.8.0 with 232 resources across 57
languages; structural validation green; the v1.8.0 release (404 assets,
digest-verified staging) publishes once the branch lands. Landing is
chained behind the plan-22 LFS reconciliation: the release is created at
the branch tip, assets upload (quota-free), URL probes go green, checks
gate the merge.

## Problem

Everything since plan 20 has shipped mirror-only pending "the owner's
cut": the fleet retrains (ca/hu/uk/vi, plan 21 A4) and the three Chinese
variants (zh-Hans-CN/zh-Hant-TW/zh-Hant-HK). Meanwhile the script-mixed
`zh` — superseded by the three variants — still occupies registry slots.
The cut resolves both in one version bump.

## What

1. **Retire `zh`**: `models/zh/` (11 files incl. its int4/nested
   experiments) and the 5 `zh.*` eval reports leave the tip (forward-only
   commit; history and the v1.7.0 assets remain). manifest drops its 6
   entries; the registry drops its 3 resources. Consumers migrate to the
   three variants.
2. **Promote 7 languages** to v1.8.0 release primaries: ca, hu, uk, vi
   (fleet retrains) + zh-Hans-CN, zh-Hant-TW, zh-Hant-HK. Every released
   full tier is release-only (plan 22 rule: mirror null); zh variants'
   small tiers keep media mirrors (they remain LFS-committed).
3. **Rebuild**: `generate_manifest.rb` CANNOT be re-run after the strip
   (it is tree-derived and would drop the 55 release-only full-tier
   entries — the validator's ground truth). The manifest is updated
   surgically: HEAD's manifest minus the 6 zh keys. This tool/tree
   divergence is now a standing property of the release-only model.
4. **Registry**: `build_registry.py --tag v1.8.0` (no `--unreleased`):
   232 resources, 57 languages, zero mirror-only full tiers, tag v1.8.0,
   all primaries at v1.8.0. Structural validation: 0 errors, 0 warnings.
5. **Release v1.8.0**: 404 assets staged and digest-verified (382 carried
   from v1.7.0 + 21 zh-variant assets from the tree + fresh
   registry.json + manifest-v1.8.0.json). Uploaded after the branch push
   (the release tag sits at the branch tip, so the tag and the merged
   main carry identical trees).

## Consumers

- **gem / kotoshu-py / rs / wasm**: `setup(:zh)` now resolves to no
  registry entry (legacy-path fallback or a clear error) — users migrate
  to `zh-Hans-CN` / `zh-Hant-TW` / `zh-Hant-HK`. The six promoted
  languages resolve to v1.8.0 primaries.
- **Browser tiers**: unchanged surfaces — mini/fluency/buckets keep media
  mirrors for every LFS-committed tier; the zh variants' tiers are
  browser-served from the moment the LFS objects land.
- **Next arc**: the gem's plan 147 (BCP-47) makes `zh-Hant-TW` vs
  `zh-Hant-HK` first-class codes end to end; the dictionaries repo needs
  Traditional dictionary packs for the variants.
