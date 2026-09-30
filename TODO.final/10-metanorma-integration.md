# 10 — Metanorma integration: spell-check at the compilation step

Kotoshu becomes the spelling engine for Metanorma, the standards
document authoring pipeline (ISO, IEC, BIPM, OGC...). Thousands of
standards documents pass through `metanorma compile` — each one gets
kotoshu's 16-language spell checking, realword detection, and
cross-script retrieval at compilation time.

## Why the compilation step

- The semantic XML is the canonical document model — all flavors
  (ISO, IEC, BIPM) produce it. Text extraction is clean.
- The document language is known from the bibdata/relaton metadata.
- Errors are actionable: reported by clause ID, before publication.
- The pipeline already has an information-extraction hook
  (sourcecode, image, requirement) — spell-check follows the same
  pattern.

## Architecture

```
metanorma compile --spell-check FILE.adoc
                    │
                    ▼
         generate semantic XML
                    │
                    ▼
    extract text runs (per clause, per language)
    using Metanorma::Document::PlainText
                    │
                    ▼
         kotoshu suggest(word, language:)
         (the engine: gem 1.0.8 — 16/16 wins)
                    │
                    ▼
         report / SARIF / annotations
```

## The three tiers

### Tier 1: `metanorma spell-check FILE` (the CLI subcommand)

In `metanorma-cli` (Thor-based):

- New file: `lib/metanorma/cli/commands/spell_check.rb` (the pattern
  is `commands/diff.rb` — plain class with exit codes)
- New subcommand: `spell-check FILE [--format text|json|sarif]`
  `[--lang LANG]` (auto-detected from bibdata if omitted) `[-o FILE]`
- Compiles to semantic XML (reusing Compile.generate_semantic_xml),
  walks the XML, extracts text runs, runs kotoshu, reports.
- Dependency: `gem "kotoshu", "~> 1.0"` (optional dependency — the
  subcommand loads lazily; compile works without it).

### Tier 2: `metanorma compile --spell-check FILE.adoc`

A compile-time gate: after semantic XML generation (Step 1), before
output generation (Step 5):

- Walk the semantic XML text runs.
- Run kotoshu per paragraph (the document language from bibdata).
- Report errors as compilation warnings (default) or fail the build
  (`--strict-spell-check`).
- The gate is opt-in; no behaviour change without the flag.

### Tier 3: SARIF output

Kotoshu results in SARIF 2.1.0 (the same output the gem already
supports for `metanorma check`): rules for nonword (suggest fix) and
realword (flag, suggest context-appropriate word).

## The dependencies (all released)

| gem | version | role |
|-----|---------|------|
| kotoshu | ~> 1.0 (1.0.8) | the engine: suggest, correct, realword, cross-script |
| metanorma-cli | (existing) | the Thor CLI surface |
| metanorma | (existing) | the compile pipeline (semantic XML) |
| metanorma-document | (existing) | PlainText extraction |

## The files

| repo | file | what |
|------|------|------|
| metanorma-cli | lib/metanorma/cli/commands/spell_check.rb | the spell-check subcommand |
| metanorma-cli | lib/metanorma/cli/command.rb | `subcommand :spell_check, ...` |
| metanorma | lib/metanorma/compile/spell_check.rb | the compile-time gate |
| metanorma | lib/metanorma/compile/compile.rb | the hook (after semantic XML) |
| kotoshu | (existing) | the engine, 1.0.8 |

## Implementation order

1. **Tier 1** (the CLI subcommand) — standalone, no compile changes,
   immediately useful for authors.
2. **Tier 2** (the compile gate) — the production integration.
3. **Tier 3** (SARIF) — the CI/IDE integration.

## What this unlocks

- Authors see spelling errors during compilation, not after.
- Language detection from the document metadata (bibdata), not
  per-file configuration.
- The full kotoshu capability column applies: realword detection,
  vocalized Arabic/Hebrew, variant-pure Chinese, cross-script
  retrieval — all at compilation time, for every standards body
  using Metanorma.
