# Plan D8: ecosystem dead-URL sweep — no consumer left behind

## Status

executed (2026-09-19) — 12 repositories swept: 11 clean (registry-driven), 1 latent dead fallback found in the gem (source_registry.rb media-host model URLs + its spec) with the exact recommended fix recorded; impact contained to the legacy fallback behind the working registry primary. Evidence: eval/reports/d8-sweep.md. Gem-side fix left to its own plan (in-flight plan 147 collision risk).

## Problem

Eliminating Git LFS orphaned every `media.githubusercontent.com/media/
kotoshu/models-fasttext-onnx` URL in existence — the LFS store is empty
by design. The registry is clean and the repo is clean, but any consumer
outside this repository that hardcodes the old media-host model URLs now
permanently 404s: the kotoshu gem's legacy download paths, the wasm/js
packages, the site, the rs/go/vscode/lsp consumers. "All things work"
must include them.

## What

1. Sweep every kotoshu repository for dead model URLs
   (`media.githubusercontent.com` references to this repo's model paths)
   and for stale LFS assumptions.
2. Verify the gem's download chain end to end against the new
   architecture: registry fetch (raw main) → primary release URL →
   fallback mirror (null-safe) → the legacy full-tier path.
3. Fix what lives in this repository and in the demo; consumer repos
   with dead URLs get reported with exact locations (they are separate
   repositories — fixes land via their own PRs, owner-gated).
4. Freeze the sweep as evidence.

## Consumers

- gem, py, js, rs, go, vscode, lsp, server, site — every model consumer.
