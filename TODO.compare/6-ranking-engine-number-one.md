# Plan C6: ranking engine — number one across the board

## Status

partially executed (2026-09-21) — gem PR kotoshu/kotoshu#226 lands the ranking path:
language_code threaded into Generator; suggest always uses the Ruby pipeline;
SymSpellStrategy is DEFAULT primary, indexes frequency full_list when present,
ranks by (distance, frequency rank); dual-layout keyboard_penalty (native+QWERTY);
Chinese IME layouts (Pinyin/Jyutping/Cangjie/Sucheng). Offline de nonword probe:
SymSpell+wiki-freq top-1 **74.7%** beats field SymSpell 73.4%. Remaining: publish
wiki-unigram Kelly JSONs to frequency-list-kelly (de/es/fr/pt/…), freeze C1 harness
reports under the fixed path, extend KELLY_LANGUAGES. C7 keyboard plan parallel.

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
