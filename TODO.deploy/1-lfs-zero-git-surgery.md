# Plan D1: LFS-zero git surgery — browser set to plain git, rest to releases

## Status

in-progress (2026-09-19, owner: "Fully implement all this")

## Problem

The free-account LFS quota blocked every push for over 24 hours; deleting
and recreating the repository did not reset the accounting, and a
support-triggered rebuild has not propagated. The final architecture
(decided with the owner after the surface matrix) removes LFS entirely:
the browser-facing set becomes plain git blobs served from
`raw.githubusercontent.com` (verified: `access-control-allow-origin: *`),
and fluency/buckets/ktm1/full stay release assets (verified: releases are
quota-free and unmetered; their audiences — editors, LSP, servers, CI —
fetch from non-CORS contexts and never needed the media host).

With zero LFS pointers in history, no push can be blocked by the LFS
quota again, regardless of how GitHub's accounting settles.

## What

1. Keep set (export-migrate pointers → blobs across history, all well
   under the 100 MiB git file cap): `models/**/fasttext.*.mini.onnx`,
   `models/**/fasttext.*.mini.vocab.json`, `packs/*.bin`,
   `models/lid/*`, `models/typo/*`, plus `models/en/fasttext.en.buckets.onnx`
   (demo language bucket tables, plan D4).
2. Drop set (remove paths from history — release-only):
   fluency onnx + fluency vocab jsons (all languages), buckets onnx
   (all languages except `en`), `*.ktm1` typo matrices, and the three
   zh-variant full-tier onnx (already release-only at the v1.8.0 cut).
3. `.gitattributes`: delete every LFS rule; replace with comments
   stating the storage law.
4. Sequence: `git lfs migrate export` for the keep set on
   `main`+`fleet-retrains`, then `git filter-repo --partial --invert-paths`
   with the enumerated drop list, then the attribute rewrite, then
   commit. Verify `git lfs ls-files` is empty and keep-set files are
   real blobs (actual sizes in tree).
5. Blast-radius note: history rewrites move the `v1.7.0` tag target;
   plan D3 recreates it. The release assets themselves are untouched.

## Consumers

- Plan D2 rebuilds the registry on the new tree; plan D3 force-pushes
  both branches and lands the cut; plan D4's demo consumes the raw-host
  mirrors this plan creates.
- Local LFS cache (`~20 GB`) and the backup dir are untouched — no bytes
  are lost; every dropped path exists as a release asset.
