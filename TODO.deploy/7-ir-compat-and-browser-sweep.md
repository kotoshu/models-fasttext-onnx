# Plan D7: IR compatibility everywhere + the 57-language browser sweep

## Status

executed (2026-09-19) — IR census: minis 57x IR10, lid migrated 11->10, en buckets RESTORED (a regression: the pack rebuild had deleted the tracked file and the deletion rode along in the IR commit - the mirror 404d until now) and migrated, typo already IR 8; every ONNX emitter pins ir_version=10; the 57-language headless browser sweep PASSed 57/57 with zero page errors; URL sweep green 465/465; README.adoc rewritten to the raw-host model; the IR floor is an AGENT-PROMPT rule. Server-set (full/fluency/buckets release assets) stays IR 11 - release-only consumers run onnxruntime >= 1.23; recorded as a follow-up, not a live defect.

## Problem

The E2E audit's fourth fix (IR version 11 vs onnxruntime-web) was applied
to the mini tiers only. Three gaps remain: (1) the other browser-served
ONNX artifacts — lid.176, typo.biencoder, en bucket tables — may still
declare IR 11 and would fail identically in any browser consumer; (2) the
converter and the other ONNX emitters still write IR 11 at the source, so
every future export re-introduces the defect; (3) the browser proof
covered three languages — the directive's bar is all 57.

## What

1. **IR census**: read the declared IR version of every ONNX file we
   ship (tree + release set) and migrate every browser-served one to
   IR 10 with sha/registry/release refresh.
2. **Root cause**: pin `ir_version = 10` in every ONNX-emitting script
   (fasttext_to_onnx.py, build_lid.py, typo trainers) so exports are
   correct at the source; record the policy as a rule (check bytes
   `08 0a`, not `08 0b`, until onnxruntime-web accepts IR 11).
3. **Server-set assessment**: full/fluency/buckets are release-only and
   load in onnxruntime >= 1.23 (proven); check the gem's onnxruntime
   expectation and record the full-set migration decision honestly.
4. **57-language browser sweep**: headless chromium drives the live
   explorer for every supported language — load model + vocab through
   the registry mirror, run a neighbour query, assert a sane answer.
5. README.adoc storage-model section updated (it still describes LFS);
   RELEASING.md checked for staleness.

## Consumers

- The audit report gains the sweep table; the rules and converter pin
  prevent the IR class of failure permanently.
