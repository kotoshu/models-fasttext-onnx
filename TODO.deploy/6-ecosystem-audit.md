# Plan D6: full ecosystem audit — every language, every surface, end to end

## Status

executed (2026-09-19) — URL sweep: 465/465 reachable; per-language audit: 57/57 PASS (byte-exact mirrors, onnxruntime load, deterministic+distinct+self-retrieval); aux artifacts PASS (lid-176, typo bi-encoder, en buckets, 3 packs); the demo language explorer is live at www.kotoshu.org/models-fasttext-onnx/realword/. Evidence: eval/reports/audit-v1.8.0.json.

## Problem

The new architecture (D1-D5) was proven on English. The owner's bar is
higher: every supported language's browser artifacts must demonstrably
work through the real public chain — registry → raw-host mirrors and
release primaries — with byte-exact downloads, loadable models, and
sane inference. An audit that only probes HTTP 200s proves nothing about
usability; one that loads nothing proves nothing about serving.

## What

1. **URL sweep**: `validate_registry.py --check-urls --urls-ref main`
   after the 410-asset upload — every non-null primary (release), mirror
   (raw host), and vocab URL probed with size agreement against the
   registry.
2. **Per-language browser audit (all 57 languages)**: for each language,
   fetch the mini onnx via its registry mirror, verify sha256 against
   the registry, fetch the mini vocab, load the model with onnxruntime,
   embed two distinct vocabulary entries, and assert: identical vectors
   for the same entry, distinct vectors for different entries, and
   self-retrieval (the nearest neighbor of entry k among a sampled
   subset is k itself). Language-independent by construction.
3. **Auxiliary browser artifacts**: LID (176-way language identification
   spot-checked on sample sentences across scripts), the typo
   bi-encoder (loads, embeds), the en bucket table (loads, bucket_ids
   tensor present), and the three packs (framing parses, sections
   decode, sha matches).
4. **Website demo, all languages**: extend demos/realword with a
   language picker over every supported language — picking one loads
   that language's mini model through its registry mirror and answers
   nearest-neighbor queries in the browser. The confusion-detection
   mode stays en-only (only en has ctx tables + a confusion table) and
   is labeled as such.
5. Findings are fixed, not just recorded.

## Consumers

- The audit table (per-language verdicts) is frozen evidence in
  `eval/reports/audit-v1.8.0.json`.
- The demo's language picker is the visible proof for every language.
