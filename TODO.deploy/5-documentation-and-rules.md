# Plan D5: documentation and rules — the storage law on record

## Status

in-progress (2026-09-19)

## Problem

The storage model changed fundamentally and must be recorded where every
future session reads it — otherwise the campaign will drift back toward
LFS (or >100 MiB git files) and repeat this week.

## What

1. `AGENT-PROMPT.md` (standing rules): add the artifact storage law —
   zero Git LFS, ever; release assets for server/CLI artifacts and
   anything ≥100 MiB; plain git served from the raw host for the
   browser set (mini, vocabs, packs, lid, typo, demo-language buckets);
   the raw host is the CORS surface, releases are not.
2. Repo docs: a storage-model section in the models README covering the
   surfaces, the mirror rule, and the free-account constraints that
   shaped them (LFS quota blocks pushes; deletion does not reset it).
3. `.gitattributes`: comments stating the law where the LFS rules used
   to be (done in D1, cross-checked here).
4. Auto-memory: round entry with the final architecture, the evidence
   (CORS verifications, quota incident), and the growth path (dedicated
   Pages-served repo per path if browser traffic ever demands it).
5. The plan-22/23 docs get a closing amendment pointing at TODO.deploy
   as the superseding record.

## Consumers

Every future working session in every kotoshu repo.
