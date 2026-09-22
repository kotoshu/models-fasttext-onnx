# Plan S7: retrieval grounding — suggestions from the user's own world

## Status

researched and scoped (2026-09-22). Retrieval-Augmented Spelling Correction for E-Commerce (arXiv Oct 2024) and To RAG or Not to RAG (SIGIR eCom) show retrieved domain context improves correction in production beyond frequency priors; QSpell-scale data work confirms the pattern generalizes across languages.

## Problem

A user's valid vocabulary is not the world's frequency list: product names, people, project jargon, personal words. Today only a flat personal dictionary exists (checked words are accepted). That is retrieval without ranking.

## What

1. Personal/corpus index: embed the user's own documents (or declared corpora) into a compact vector sidecar; corrections matching retrieved context get a grounding boost in the composite ranking (a feature, not a gate — precision law).
2. Domain-pack shape: a kotoshu domain pack (glossary + optional vectors) loads per project — the libraries-have-no-side-effects law applies (packs are data, read-only, consumer-side paths).
3. Eval: build a personal-domain eval split (typo'd project jargon) through the C1 harness; measure lift vs the flat personal dictionary.
4. Privacy posture: retrieval runs fully local; nothing leaves the machine (the client-server lanes of S6 carry only anonymous slates).

## Consumers

Editor/browser surfaces; the domain-pack ecosystem; the "content quality checker, not just typo checker" mandate.
