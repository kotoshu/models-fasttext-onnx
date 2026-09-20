# Plan D12: continuous verification — the arc watches itself

## Status

executed (2026-09-20) — workflow .github/workflows/daily-verify.yml live (cron 06:23 UTC + manual dispatch); first dispatch GREEN: manifest tree-parity OK (176 match, 220 exempt), URL sweep 465/465 reachable, release digest spot check matches the registry. TODO.deploy closes with this plan.

## Problem

Everything built in TODO.deploy/1-11 is green today, but nothing
re-verifies it after the fact: the served registry, the raw-host
mirrors, the release asset digests, and the surgical manifest can drift
(a file moved, an asset deleted or replaced, a tag re-pointed) with no
push to trigger CI. The D9 lesson — every bypassed gate accretes silent
debt — applies to time, not only to PRs.

The sixth audit found no defects. This plan adds the one real
deploy-domain gap that remains and then closes the domain.

## What

1. A daily scheduled workflow on the models repository (plus manual
   dispatch) that runs, against the LIVE state:
   - the manifest tree-parity check (surgical manifest vs tree truth)
   - the full registry URL sweep at main (primaries, raw mirrors,
     vocab URLs, size agreement)
   - a release-asset digest spot check: three representative v1.8.0
     assets (raw-served mini, release-only fluency, ktm1 matrix)
     compared against the registry's declared sha256
2. A red run is the alert; no issue automation (the org watches CI).
3. Verify the workflow green via manual dispatch immediately.

## Consumers

Future drift of any serving surface becomes visible within a day
instead of at the next user report.

## Domain closure

After D12, TODO.deploy is complete: architecture, migration, registry,
demo, audits (URL, language, browser, consumer, dependency), releases,
CI, and monitoring. Remaining campaign work is implementation domain
(TODO.impl): the ctx-LM Phase-1 gate, per-language confusion/context
tables for all-language detection, the Korean subword floor, the
Japanese retrain, fa/lv/pl/sv second sources, gem plan 147.
