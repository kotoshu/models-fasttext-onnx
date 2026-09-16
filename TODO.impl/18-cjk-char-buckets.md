# Plan 18: CJK char-level bucket feasibility (real-word detection, CJK angle)

## Status: feasibility probe EXECUTED (zh) — retraining solves row starvation, but the sibling-artifact architecture fails structurally; verdict recorded, owner fork

Findings (evidence: eval/reports/zh.buckets.char-probe.json, harness:
scripts/train_char_sibling.py — reproduces every number below):

1. ROW STARVATION SOLVED. A sibling fastText (jieba-segmented
   Wikipedia shard, minn=1/maxn=3, bucket=200k, 102 s train) yields
   79,085/200,000 trained bucket rows where the Common Crawl binary
   starves short-token rows. The plan's core hypothesis is confirmed.
2. CROSS-SPACE COMPOSITION IS THE REAL BLOCKER. The bucket exporter
   composes sibling bucket rows against CRAWL tier rows and ranks
   against the crawl vocab — two different embedding spaces. Even the
   full-table reference scores intended_top1 = 0.0000. A Procrustes
   alignment over 7,765 shared-vocab anchors transfers weakly
   (relative residual 0.785) and post-alignment signal is thin
   (intended_top5 16.7% on 18 usable demand probes). A "sibling OOV
   artifact" bolted onto crawl tiers cannot work as specced.
3. THE zh GATE IS CORPUS-STARVED. The zh typo corpus supports 3
   non-overlapping gate probes (en: 400) and 18 in-tier demand probes
   — no statistically meaningful pass is possible for zh regardless
   of scorer. The 8 bucket rejections were never only a subword-range
   problem.

NEXT RUNGS (owner fork): (a) same-space route — a jieba-segmented zh
model serving BOTH tier rows and bucket rows (a zh model replacement
arc through the existing tier gates; heaviest but clean); (b)
alignment route — research-grade cross-space alignment (CCA, more
anchors, mixed-script handling) keeping the sibling as OOV-row
source; (c) close plan 18 as infeasible-as-specced. All three need a
larger CJK typo corpus first (the shared blocker with the KTM1 arc).

## Problem

Eight languages are gate-rejected for buckets (ar, cs, fa, he, ja, pl,
vi, zh) per the Phase-0 evidence; the rejection rationale in
eval/reports/{lang}.buckets.json is that the Common Crawl fastText
binaries are trained with `minn=maxn=5`, so character n-grams shorter
than five are never stored in the model's input matrix and the bucket
table (which reads stored rows verbatim) is empty for short tokens.
CJK words are typically 1-3 characters and the rejection follows from
that.

That reasoning assumes the OOV table MUST come from the same model's
subword scheme. It does not: a separate small char-level OOV model
trained per language with `minn=1` (or 2-4 for sub-character
robustness) provides rows at every length and the bucket export runs
against THAT model's stored rows. The char-level model's tier
embeddings stay separate from the main `fasttext.{lang}.onnx` - it
ships as a sibling artifact for the buckets resource only.

## What (the proposal; the probe decides whether it is feasible)

- Train a per-language fastText skipgram on the same Wikipedia shards
  already fetched (plan 16 infrastructure), with `minn=1/maxn=3`
  subwords. Local training first (char-BiGRU precedent: ~12 CPU-min
  per language); the Modal recipe exists for the multilingual case.
- Export the bucket table from THIS model via the existing
  `scripts/export_buckets.py` (the size/lift-over-baseline/fidelity
  gates apply unchanged).
- If passes for zh first: ship `fasttext.zh.buckets.onnx` + registry
  entry, repeat for the other 7 languages in corpus availability
  order.
- The detector (plan 17 / 146 / 07) gains a useful CJK vocabulary.

## Feasibility probe (cheap, decisive, no commitment)

`scripts/export_buckets.py --lang zh --bin downloads/<char-level.bin.gz>
--minn 1 --maxn 3` against the plan-16 infrastructure: does ANY
bucket row get produced at all for CJK, and does the size / lift /
fidelity gate pass for at least one tier? Numbers, not assumptions.

## Consumers

Real-word detection for CJK is the gap that plan 17 explicitly leaves
open (homophone tables vs n-gram/transformer). Bucket OOV coverage is
independent of detection; it improves the existing semanticSuggest
story (plan 9) for those languages regardless of where the detection
arc lands.
