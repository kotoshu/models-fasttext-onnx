# Plan C4: grammar and style — the gap map to "content quality checker"

## Status

scoped and measured (2026-09-21) — the inventory is THREE English rules total (EN_A_VS_AN, EN_THERE_THEIR, EN_DOUBLE_NEGATIVE) over 8 pattern-matcher classes (rule_engine/rule_loader + YAML rules in the dictionaries repo, en/grammar/rules.yaml); the engine is open/closed and the gap is rule CONTENT, not plumbing. LanguageTool-class coverage means thousands of rules per major language (per their published docs; no precise count asserted here) across grammar, style, register, and typography categories. The honest path: (1) a rule-authoring harness with per-rule FP budgets (the typo arc's lesson - never ship a rule on vibes); (2) priority from C1's error-class breakdown; (3) per-language batches. This is a multi-quarter content program - this plan records the map and recipe; it does not pretend to ship it in one pass.

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
