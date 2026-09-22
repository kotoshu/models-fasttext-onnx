# Impl S6: quantization + latency lanes (graph node S6)

## Status

pending models. Deliverables: `scripts/quantize_lane.py` (onnxruntime int8 dynamic/static quant, per-lane size/latency measurement); lanes recorded in model manifests (mini ≤5MB/≤10ms, fluency ≤20MB/≤25ms, server ≤100ms/sentence); conformance replay per tier.

## Gate

G-S6: measured budgets in manifests; replay green.

## Consumers

Every neural artifact (existing fleet + S2/S3).
