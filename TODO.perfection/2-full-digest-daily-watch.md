# Plan P2: full-digest daily verification

## Status

executed (2026-09-20) — the daily watch now digest-verifies every registry-pinned v1.8.0 release asset; first dispatch green (465/465 URLs + tree parity + full digests).

## Problem

The daily watch spot-checks three release assets. Perfection means the
daily run verifies EVERY v1.8.0 release asset digest against the
registry's declared sha256 — drift on any of the 410 assets surfaces
within a day.

## What

The daily-verify workflow's digest step compares all v1.8.0 assets
against the registry's declared sha256 (with a tolerance for the
manifest-v1.8.0/registry.json assets whose sha lives in the release
metadata itself). Registry-declared entries get byte-exact comparison.

## Consumers

The architecture's continuous watch becomes total.
