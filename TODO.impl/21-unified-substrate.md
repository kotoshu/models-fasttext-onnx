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

## The gates (CORRECTED 2026-09-17 per owner: quality over coverage)

OWNER LAW: "bad input quality gives bad output quality" - the
register-diversity gate as originally drafted was WRONG. The remedy
for a vocabulary gap is never raw web text; it is MORE QUALITY DATA:
quality sources that span usage (edited news and published books
cover modern colloquial vocabulary at editorial quality). Register
!= quality.

1. TEXT QUALITY (the first gate, not one among equals): every source
   is edited/curated (encyclopedic, published books, professionally
   edited news) OR passes an explicit quality-filtering pass
   (C4-style heuristics + dedup + cleanliness measures) - and the
   bar is judged on OUTPUT evals (tier rank-corr, bucket
   lift/fidelity, suggestion-quality benchmarks), not asserted.
2. VOCAB COVERAGE >= 99% of the per-language reference wordlist
   (the existing cc.*.300 vocab for its script, or a national
   frequency dictionary) over the reference's top 20k - achieved by
   adding more QUALITY sources (cost no object), never by relaxing
   the quality bar. The plan-20 Wikipedia-only model fails at 85.7%
   and stays a pipeline proof; the FIX is quality sources spanning
   usage, not crawl noise.
3. DEDUP + script purity (the plan-18 separation rule) + train/eval
   contamination checks.
4. COMPACTNESS: the existing tier size budgets unchanged (int8
   per-row, ~15MB fluency / ~3MB mini).
5. The per-check gates (bucket lift/fidelity, real-word split, tier
   rank-corr) unchanged - the substrate serves the checks, and each
   check's own gate still governs its artifacts.

Consequence for the zh-Hans retrain: the corpus is quality-first -
Wikipedia + edited news + published books - with a quality-filtered
crawl pass ONLY if the coverage gate still fails and the pass clears
the quality bar. "Crawl scale" alone is no longer the target;
"quality scale" is.

## Execution sequence (highest measured gap first)

1. zh-Hans-CN retrain at crawl scale (Chinese-Common-Crawl-Filtered,
   verified live) - replaces the plan-20 pipeline-proof model under
   the same BCP-47 code; the registry swap is a shas/pairing regen.
2. zh-Hant (CC-100 zh-Hant, verified live) with TW/HK vocabulary
   separation downstream (per the three-variant directive).
3. ja retrain (minn=1/3 + crawl scale - its bucket gate failed on
   subword starvation, the same defect class as zh).
4. ko + the 55-language fleet audit against the quality + coverage
   gates: retrain only the languages that FAIL; small languages may
   keep Wikipedia-base where it wins the measurement; the cc.*.300
   legacy (raw 2018 Common Crawl) is tolerated until the audit
   measures whether a quality-first retrain beats it on output evals.

## Regional variants (owner 2026-09-17: "clearly US english and UK
english are different as well")

The variant architecture is GENERAL: BCP-47 codes cover regional
variants, and the plan-19 schema already accepts them (en-US, en-GB,
pt-BR, pt-PT, de-CH, fr-CA... all validate today). The separation
PRINCIPLE differs by what actually differs:

- SCRIPT forces model separation (zh variants: different characters +
  vocabulary -> separate substrates, the plan-18 rule).
- REGION within a script usually forces DICTIONARY/CHECK separation
  only: en-GB vs en-US is spelling conventions (colour/color,
  -ise/-ize, travelled) + vocabulary use + punctuation norms - the
  SUBSTRATE (vector semantics) is shared; the hunspell dictionaries
  (en_GB/en_US both exist upstream) and the regional check config
  carry the variant. A shared-substrate decision must still pass the
  coverage gate per variant.
- The variant ledger (initial): en-US + en-GB (first pair; the
  dictionaries exist upstream), pt-BR + pt-PT (the spelling reform
  makes them strongly distinct), de-DE/de-AT/de-CH (ss/esszet),
  es-ES + es-419, fr-FR/fr-CA. Each lands as dictionary-pack +
  registry codes; substrates split only where the gates demand.

The gem half (plan 147) extends accordingly: variant resolution maps
a BCP-47 code to (dictionary pack, regional check config, substrate
model) - substrate shared by reference unless separated.

## Consumers

Plan 22 (the checks framework) rides the substrate: spelling, typo
retrieval, real-word (plan 17), fluency-as-a-check, grammar. The
substrate is the shared embedding bed; the checks are its customers.
