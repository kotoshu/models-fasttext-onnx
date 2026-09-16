# Plan 18: CJK char-level bucket feasibility (real-word detection, CJK angle)

## Status: proposed - a feasibility probe is cheap and worth running before owner sequencing

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
