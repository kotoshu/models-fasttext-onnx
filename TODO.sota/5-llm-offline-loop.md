# Plan S5: LLM strictly offline — teacher, judge, hard-negative miner

## Status

researched and scoped (2026-09-22). The literature is unambiguous that LLMs belong OFFLINE for our product class: over-correction destroys F0.5 (Fang arXiv:2303.14342; Lin 2024 cited 20; ACL 2024 realistic-CSC benchmark; edit-level majority voting arXiv 2026), while "Adapting LLMs for Minimal-edit GEC" (arXiv June 2025) and Google's Distilling Step-by-Step show LLMs excel as TEACHERS for small minimal-edit students. Grammarly's production shape (1B on-device) and Gboard Proofread (arXiv June 2024, server LLM) confirm: LLM at training/serving-heavy tier, small model at inference.

## Problem

We need more training data and honest evals; LLMs are the cheapest strong source of both — if and only if they never touch the suggestion hot path (precision law; C1 harness measures the small models, not the API).

## What

1. Teacher distillation: prompt a strong LLM for minimal-edit corrections on our clean corpora → (corrupted, corrected) pairs filtered by edit-level agreement (majority voting) → S2/S3 training data.
2. Hard-negative mining: LLM proposes plausible-but-wrong corrections for real-word sentences → negatives for the S3 reranker.
3. LLM-as-judge for eval hygiene: audit our frozen splits (label noise estimation) and grade wave-2 generated splits (plan S1/S8) — reported as inter-annotator agreement, never as a product lane.
4. Cost discipline: bounded budgets, cached outputs committed as artifacts (no live API in CI hot loops).
5. The product's runtime stays: SymSpell + rankings + ONNX S2/S3. Zero API dependency.

## Consumers

S2/S3 data quality; wave-2 split trust; the honest-claims law (field lanes measure real systems).
