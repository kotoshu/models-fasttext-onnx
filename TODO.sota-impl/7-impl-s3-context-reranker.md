# Impl S3: contextual MLM reranker (graph node S3)

## Status

pending S1 sentence pairs. Deliverables: S1 extension emitting (sentence, slot, candidates, truth) quads from clean corpora; MiniLM fine-tune (masked-slot scoring); ONNX int8; check-path optional rerank in the gem (context threading through the checker); C1 realword sentence-slice gate.

## Gate

G-S3: realword lift with FP budget unchanged; ≤10ms/sentence server lane.

## Consumers

Real-word class in the check() path (where context exists).
