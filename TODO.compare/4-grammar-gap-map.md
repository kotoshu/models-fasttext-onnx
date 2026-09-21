# Plan C4: grammar and style — the gap map to "content quality checker"

## Status

scoped (2026-09-21) — measured gap map, no fake implementation

## Problem

The product is a content quality checker, but the Checks framework
ships exactly two checks: spelling (default-on) and grammar (opt-in,
en-only rule set). LanguageTool-class coverage is the structural gap
between spell checker and quality checker.

## What

The honest map, recorded here, with the real state measured:

1. Current inventory: the gem's Grammar::RuleEngine rule count, the
   languages covered, the pattern-matcher classes — measured from the
   code, not assumed.
2. The field's shape: LanguageTool's rule taxonomy (grammar, style,
   register, clarity) as the comparison frame; the gap stated per
   category.
3. The architecture verdict: the Checks framework (Registry, Base,
   Finding) was built open/closed for exactly this — new checks are
   new classes, not core edits. The gap is rule CONTENT, not plumbing.
4. Prioritization input: the C1 benchmark's error-class breakdown
   says which error classes cost the most accuracy — grammar rules
   chase those classes first, per language.

Rule authoring is multi-week content work with its own eval harness
needs (false-positive budgets per rule, as the typo arc taught);
this plan records the map and the recipe, and does not pretend to
ship rules in one pass.

## Consumers

The Checks framework roadmap; the C5 per-language evals quantify the
error classes that grammar rules should chase.
