# Plan 09 (models) / 107 (corpus) — Bucket tables for every model language

## Why
Plan 103 proved bucket-backed OOV (Teh -> the from the model path) and
shipped en/de siblings; 53 model languages still embed OOV words only
through vocab-present n-grams. The export pipeline
(scripts/export_buckets.py, K=32768 usage + typo-corpus demand) is
mechanical; the gap is data, not code.

## Work (models repo)
1. Run the export for every registry language lacking a buckets sibling
   (53 after en/de), in batches that keep each push reviewable; commit
   the .onnx via LFS (mirrors must resolve), extend each language's
   tiers.json buckets entry, regenerate manifest.json.
2. Registry v1.5.0: additive kotoshu://models/{lang}/buckets entries,
   release-tagged primary URLs, media-host mirrors; validator already
   accepts the shape.
3. Release v1.5.0 attaching every new bucket asset.
4. Size discipline: if any language's trimmed table exceeds ~15 MB,
   reduce K for that language and record the choice in its tiers.json.

## Verification
validate_registry --check-files green; mirrors spot-checked 200; release
assets all 200; eval probe on 3 sampled languages shows the OOV-resolved
fraction improving as en did.

## Status
Pending
