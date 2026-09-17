# Plan 20: the zh-Hans-CN same-space model (plan 18a execution)

## Status: in progress - corpus segmented, training launched

The owner opened every gate on 2026-09-17: BCP-47 codes shipped (plan
19), distribution approved, scrape authorized. This plan builds the
first variant model end to end — zh-Hans-CN — as the template the TW
and HK builds repeat.

## Pipeline (each stage gates the next)

1. **Corpus**: scripts/segment_variants.py over Wikipedia shard 0 →
   eval/corpus/zh-hans-wiki.txt — DONE: 268 MB, 1,242,757 lines
   (2,315,513 Traditional lines went to zh-hant-wiki.txt instead;
   241,618 mixed lines dropped from both).
2. **Train**: fastText skipgram, dim=300, minn=1/maxn=3 (the setting
   that solved CJK row starvation), bucket=2M, minCount=5, epoch=5 —
   the SAME bin serves tiers AND buckets (same space by construction;
   the crawl-model cross-space failure cannot recur).
3. **Tiers**: dump .vec → scripts/build_tiers.py — the full/fluency/
   mini ladder with its existing gates (fluency rank correlation,
   mini top-1 agreement) unchanged.
4. **Buckets**: scripts/export_buckets.py --bin <the new bin>
   --corpus eval/corpora/zh-csc.json (19,212 real CSC pairs) — same
   space, real demand signal; all gates unchanged.
5. **Registry**: entries under zh-Hans-CN (plan 19 codes); the legacy
   script-mixed zh entry is RETIRED at the cut that ships the
   replacement (owner call at cut time).

## Consumers

The gem half (plan 147) once the model ships; the TW/HK builds repeat
this pipeline with their corpora (zh-hant-wiki.txt + vocabulary
filtering; hk-hansard text when the scrape completes).
