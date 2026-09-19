# Plan D9: consumer null-safety, cross-browser proof, CI health

## Status

in-progress (2026-09-19, fifth issuance of the standing directive)

## Problem

Three unverified layers remain. First, `urls.mirror: null` for
release-only entries reached the *served* registry (raw main) only with
the D2 commit — before that, every served registry had mirrors for all
entries. Any consumer that deserializes `mirror` as a non-optional
string (Rust serde structs, TypeScript types, Python dict access)
breaks on the live registry TODAY. The gem is proven null-safe; py, rs,
js, go are not. Second, every browser proof ran on Chromium; WebKit and
Firefox have never loaded the page. Third, the recreated repository's
CI has never been confirmed green.

## What

1. **Consumer null-safety**: inspect the registry-consuming types/paths
   in kotoshu-py, kotoshu-rs, kotoshu-js, kotoshu-go; exercise each
   parser against the live v1.8.0 registry bytes. Fix or report with
   exact patches.
2. **Cross-browser E2E**: install WebKit and Firefox for playwright;
   run the full demo flow (model chain, detection, explorer) on both.
3. **CI health**: list workflow runs on the recreated repo; if red or
   never run, diagnose and repair the workflows.
4. Evidence frozen; rules updated if a new lesson emerges.

## Consumers

py/rs/js/go packages, every browser user, and the repo's own CI.
