# Plan C6: ranking engine — number one across the board

## Status

in progress (2026-09-21). Root causes measured:

1. **Native backend short-circuit.** Default `backend: "auto"` routes `suggest` through kotoshu-rs Hunspell with `ranked: true` (verbatim order). The language-aware Ruby scorer never runs when the extension is loaded. Conformance vectors are en-only for suggest — non-English native order is ungoverned. Symptom: de junk ("ihr-t", "ihr.t") outranks "ihr".
2. **language_code never reached EditDistanceStrategy.** Spellchecker built Generator without `language_code:` → strategies defaulted to `'en'` frequency tiers + QWERTY for every language. Patched both Generator.new sites (gem, uncommitted).
3. **Candidate pool is the wrong dictionary.** Hunspell aff+dic includes compound-split artifacts. SymSpell beats us because it indexes a compact frequency list. The gem already has `SymSpellStrategy` but (a) it is not in DEFAULT_ALGORITHMS, (b) autoload constant is misspelled (`SymspellStrategy` vs class `SymSpellStrategy`), (c) it indexes the Hunspell lexicon, not the frequency full_list. Offline probe: SymSpellStrategy + wiki-freq-de full_list + (distance, −ipm) rank is the path to match/beat field SymSpell.

German Ruby-engine baseline after language_code fix: nonword top-1 54.4% (SymSpell 73.4%, Hunspell 54.4%); top-5 77.2% — correct word is usually present, order loses. Real-word: kotoshu 12.5% best-in-class.

## Problem

We must be #1 against Hunspell and SymSpell fully across the board, for every language we ship, with keyboard-aware ranking.

## What

1. Fix Generator language_code pass-through (done, needs commit + specs).
2. Fix SymSpellStrategy autoload name; add it to DEFAULT_ALGORITHMS as the primary nonword channel.
3. Feed SymSpellStrategy the frequency-list `full_list` when present (else Hunspell words); rank by `(distance, −frequency)`.
4. Native backend: either re-rank native rows through the Ruby scorer, or keep `correct?` native and `suggest` on the Ruby pipeline. Conformance must cover de/fr/es suggest order, not only en.
5. Publish wiki-unigram frequency JSONs (de/es/fr/pt/…) into kotoshu/frequency-list-kelly; extend KELLY_LANGUAGES.
6. Re-run C1 harness for en + de + es + fr under the fixed path; freeze reports. Target: beat SymSpell top-1 on nonword AND keep real-word lead.

## Consumers

C5 per-language evals; C7 keyboard universe (layout penalty sits on the same scorer); every product surface that calls suggest.
