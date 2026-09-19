# Plan D10: rebase-merge the rs fix, release it, and green every CI in the org

## Status

executed (2026-09-20) — PR #50 rebase-merged after its two red checks were root-caused (stale en-mini checksum pin; stale verbatim registry fixture; pack entries unparseable -> Tier became an untagged enum; the wasm worker + memory-ceiling scripts pinned the DEAD v1.5.0 registry tag, dead since the models repo recreation -> both pinned to v1.8.0, whose tag was re-pointed at the current main first). kotoshu 0.2.3 live on crates.io (tag kotoshu-v0.2.3, trusted publishing); @kotoshu/worker 0.1.1 live on npm (PR #53 added the missing fixture-sync step to its publish workflow after the first attempt failed ENOENT). hk-hansard CI red since 09-17 fixed via baseline refresh (PR #5, all findings were project vocabulary). The models release gate restructured to validate the cut (tier rebuild is impossible under the release-only model) and its notes generator taught the typo-matrix tier; the tag-triggered release workflow is GREEN. Org-wide final sweep: 12/12 repositories green.

## Problem

kotoshu-rs PR #50 fixes live registry breakage and waits on merge; the
crate needs a release carrying the fix; and the D9 lesson (silent red CI
on a bypassed path) demands an org-wide CI sweep — no repository's
latest run may be red without someone having read why.

## What

1. Read PR #50's check verdicts; rebase-merge only if green.
2. Release kotoshu-rs following the repo's own release convention and
   the workspace's declared version (never inventing a number); verify
   the published artifact.
3. CI sweep: latest run conclusion for every kotoshu organization
   repository; diagnose and fix every red one in its own repo; also
   sweep workflows for stale LFS steps (the storage model deleted LFS).
4. Evidence frozen; memory updated.

## Consumers

Every repository's CI contract; every rs registry consumer awaiting
the fix.
