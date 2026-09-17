# Plan 21: the unified semantic substrate - one approach, per-language corpora by measurement

## Status: proposed (owner directive 2026-09-17: "same semantic approach for all languages; whether wikipedia is good depends on the data; quality and compactness matter, cost does not")

## Principle

Every language gets the SAME substrate architecture - one fastText
embedding model per language, the full/fluency/mini int8 tier ladder,
the same subword policy, the same registry/pairing/bucket
infrastructure - and the corpus is chosen PER LANGUAGE by measurement,
never by dogma. Wikipedia is the right corpus exactly where it is the
best data that exists (true for small languages, false for major
ones; measured, not assumed). Training cost is explicitly not a
constraint; vocab quality and register diversity are.

## The measurement gates (a language's model ships only if all pass)

1. VOCAB COVERAGE >= 99% of the per-language reference wordlist
   (the existing cc.*.300 vocab for its script, or a national
   frequency dictionary) over the reference's top 20k words. This is
   the false-flag guard - the plan-20 Wikipedia model fails at 85.7%
   (14.3% of common words missing) and is thereby correctly held to
   pipeline-proof status.
2. REGISTER DIVERSITY: the corpus demonstrably spans wiki + news +
   web/colloquial registers (a content-quality checker sees all
   registers; encyclopedic-only models flag correct modern usage).
3. DEDUP + script purity (the plan-18 separation rule) + contamination
   checks (dedup across train/eval).
4. COMPACTNESS: the existing tier size budgets unchanged (int8
   per-row, ~15MB fluency / ~3MB mini).
5. The per-check gates (bucket lift/fidelity, real-word split, tier
   rank-corr) unchanged - the substrate serves the checks, and each
   check's own gate still governs its artifacts.

## Execution sequence (highest measured gap first)

1. zh-Hans-CN retrain at crawl scale (Chinese-Common-Crawl-Filtered,
   verified live) - replaces the plan-20 pipeline-proof model under
   the same BCP-47 code; the registry swap is a shas/pairing regen.
2. zh-Hant (CC-100 zh-Hant, verified live) with TW/HK vocabulary
   separation downstream (per the three-variant directive).
3. ja retrain (minn=1/3 + crawl scale - its bucket gate failed on
   subword starvation, the same defect class as zh).
4. ko + the 55-language fleet audit against the vocab-coverage and
   register gates: retrain only the languages that FAIL (small
   languages may keep Wikipedia-base where it wins the measurement;
   the major languages keep cc.*.300 until a gate says otherwise).

## Consumers

Plan 22 (the checks framework) rides the substrate: spelling, typo
retrieval, real-word (plan 17), fluency-as-a-check, grammar. The
substrate is the shared embedding bed; the checks are its customers.
