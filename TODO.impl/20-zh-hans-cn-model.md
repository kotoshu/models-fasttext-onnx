# Plan 20: the zh-Hans-CN same-space model (plan 18a execution)

## Status: EXECUTED - the first variant model is built and gates-passed

Results (2026-09-17, all on this machine, reproducible from the
pipeline in eval/cache/zh-hans-pipeline.sh):

- Training: fastText skipgram over zh-hans-wiki.txt (48.4M tokens,
  236,243 vocab) with minn=1/maxn=3, bucket=2M - the short-ngram
  setting that solved CJK row starvation; 535,009/2,000,000 trained
  bucket rows.
- Tier gates: fluency rank_corr 1.0000 / top1 1.0000; mini
  rank_corr 0.9999 / top1 1.0000 - above the fleet's worst-case
  thresholds (0.9998/0.950).
- Bucket gates: PASSED at K=32,768 and K=65,536 (resolved 1.000 vs
  baseline 0.000; fidelity agree/cos 1.0000/1.0000); K=131,072
  fails the size budget as the ladder intends. Artifact
  fasttext.zh-Hans-CN.buckets.onnx ships at 9.8 MB (32,768 usage rows
  + 25 demand rows from the real CSC corpus).
- Registry: regenerated at tag v1.7.0; the diff is exactly the four
  zh-Hans-CN resources (full/fluency/mini/buckets), mirror-only per
  the serves-from-main convention until a release cut carries them.

Honest caveat recorded: the bucket gate's corpus-probe side is thin
(2 probes) - structurally, CSC errors are REAL-WORD class (the typo
is in-vocab), so they do not qualify as OOV probes; the pass rides on
the synthetic ladder plus those two. The real-word-detection arc
(plan 17) is the consumer for that class, not the bucket gate.

Pipeline notes for the TW/HK builds: save_vectors is absent in this
fastText build (manual .vec dump); the manifest generator requires
git-tracked artifacts (stage before regenerating); build_tiers needs
the language in the manifest; eval/noise.py now routes the variant
codes through the CJK confusion path (zh-Hans-CN shares the
simplified pairs; the Hant variants need Traditional pairs when
built).

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
