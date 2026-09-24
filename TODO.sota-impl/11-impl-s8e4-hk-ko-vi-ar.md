# Plan S8-E4: extend the wave-2 program to zh-Hant-HK, ko, vi, ar

## Status

prep-complete (2026-09-24). All four languages have frozen splits
(2,000 nonword + 200 realword, per-class tags) and frequency lists
(kelly#8 merged: ko 26,795 / vi 10,387 / zh-Hant-HK 65,121 entries);
the jyutping table (29,936 chars from Unihan kCantonese, 100% top-10k
wordlist char coverage) is committed confusion evidence. Field dicts:
ar/ko/vi staged in the dictionaries repo; HK's kotoshu lane resolves
through the script-base fold to zh-Hant, and its field lane reads 0.0
like the other CJK (no zh hunspell exists upstream) — disclosed.
Benches queued serially behind fr/nl/pl/pt.

Generator fixes landed en route (frozen splits keep their files): the
ime-confusion class had been dead since S1 (char/syllable map
inversion); far-sub and adjacent-sub fallbacks injected Latin letters
into non-Latin scripts; diacritic-omit misfired on NFD hangul jamo;
real-word pairs are now generated constructively from the confusion
tables (the rejection sampler collapsed to single digits) with vi on
tone-stripped skeleton groups.

## What already exists (verified 2026-09-24)

- **S1 layout grids**: `Vietnamese-QWERTY`, `Arabic-101`,
  `Dubeolsik` (ko), `Jyutping` + `Cangjie` + `Sucheng` (HK IMEs),
  `Pinyin` — all present in the generator's grid set.
- **Field dictionaries** (kotoshu/dictionaries, staged codes):
  `ar` 465,928 entries; `ko` 101,598; `vi` 6,631.
  **No zh-Hant-HK.**
- **Frequency lists** (frequency-list-kelly): `ar` ✓ (kelly#7 blend,
  Kelly head + wordfreq tail). **No ko, vi, zh-Hant-HK.**
- **Models**: all four already ship (ar/ko/vi fleet retrains;
  zh-Hant-HK round #37) — nothing to build.
- **Harness**: `benchmark_suggesters.py --split2` is language-generic
  (hunspell / symspell / kotoshu lanes); `KOTOSHU_BENCH_TIMEOUT`
  env-overridable for the slow kotoshu lane; wordfreq covers ko/vi/ar.
- **IME-confusion machinery**: the generator's `ime-confusion` class is
  table-driven (char → syllable → homophones); a jyutping table drops
  in unchanged.

## Per-language gaps

| lang | split wordlist | frequency list | field dict | typo/IME classes |
|------|----------------|----------------|------------|------------------|
| zh-Hant-HK | new list | **NEW** (zh-Hant head + s2hk vocabulary anchors, wordfreq tail) | **NEW** (zh-Hant dic s2hk-converted; derivation disclosed) | Jyutping homophones (Unihan `kCantonese` — same class machinery); Cangjie/Sucheng layout classes |
| ko | new list | **NEW** (wordfreq ko) | ✓ staged (101,598) | Dubeolsik layout + jamo-composition class (NEW); eojeol unit |
| vi | new list | **NEW** (wordfreq vi) | ✓ staged (6,631) | Vietnamese-QWERTY layout + tone-mark class (NEW: dấu placement/movement) |
| ar | ✓ | ✓ | ✓ staged (465,928) | Arabic-101 layout + hamza/taa-marbuta class (NEW: أ إ آ ا، ة/ه، ى/ي) |

## Deliverables

1. **Confusion/IME tables** (same pattern as `build_confusion_tables.py`):
   jyutping homophones from Unihan `kCantonese` (license-permissive;
   coverage measured and reported before the split freeze); vi
   tone-mark placement rules; ar hamza/taa-marbuta sets; ko dubeolsik
   jamo pairs.
2. **kelly PR**: `ko.json`, `vi.json`, `zh-Hant-HK.json` + manifest
   regen (house blend shape: quality head verbatim, wordfreq tail
   appended — never interleaved; the ja junk-interleave incident is the
   standing counter-example).
3. **dictionaries PR**: `zh-Hant-HK` entry (derived, disclosed); the
   staged-codes rule says edit the dictionaries repo, not the gem list.
4. **Generator extension**: the four languages + new classes; frozen
   splits ≥2,000 nonword + ≥200 realword each, per-class tags.
5. **Serial bench runs** → frozen reports (`suggest-benchmark-{lang}-wave2.json`)
   → verdict table extension.
6. **TODO.sota-impl/2 + master-graph** status updates.

## Gates (G-S8 shape, per language)

- ≥2,000 nonword + ≥200 realword frozen pairs, class distribution
  recorded, splits reproducible corpus-free from the same generator.
- kotoshu ≥ every field lane on nonword and realword; for HK the
  hunspell lane is the disclosed s2hk-derived dictionary (or skipped,
  with the gap stated) — the SymSpell lane and kotoshu remain the core
  comparison.
- Every claim traces to a committed frozen report before the website
  announcement consumes it.

## Honesty notes (recorded, not hidden)

- **ko**: word-level (eojeol) bench underrepresents agglutinative
  spelling reality; the morpheme/subword gate stays the models-side arc
  (round #37's audit verdict stands).
- **vi**: the staged field dictionary is small (6,631) — the hunspell
  lane is weak by construction; the primary comparison is kotoshu vs
  the SymSpell lane.
- **zh-Hant-HK**: the Jyutping IME class is the dominant phone-typing
  path; Cangjie/Sucheng have layouts but no confusion tables — those
  languages' IME users are covered by layout-substitution classes only.
- **ar**: RTL text — the generator's classes operate on string indices
  and are direction-agnostic (verify once in the first generated batch).

## Risks

- ko frequency lists from wordfreq subtitles registers skew
  colloquial; the same caveat as ja (documented, not blocking).
- Unihan kCantonese coverage gaps for rarer chars — measure before the
  freeze; uncovered chars fall back to layout/far-sub classes.
- Bench cost: the kotoshu lane runs 2–6 h/language on this machine
  (ru: ~6 h); four new languages ≈ a day of serial wall time.
