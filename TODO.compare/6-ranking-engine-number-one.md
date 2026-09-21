# Plan C6: ranking engine — number one across the board

## Status

executed (2026-09-21) — gem PR kotoshu/kotoshu#226 + frequency-list-kelly#2 (merged).
**English: kotoshu is #1 on every measured slice** (2000+2000 pairs, frozen
eval/reports/suggest-benchmark-en.json): nonword top-1 86.4% (SymSpell 85.4%,
Hunspell 78.5%), top-3 94.4%, top-5 95.7%; real-word top-1 9.4% best-in-class.
Decisive fixes: true Damerau distance (not deletion-level approximation),
double-letter pattern ordering (the dominant error class), alphabetical-tiebreak
elimination (ranked:true end-to-end), Kelly+wiki blended en list (42,585 words).
**German: #1 on top-3 (91.1%), top-5 (92.4%), real-word (11.1%)**; top-1 70.9%
vs SymSpell 73.4% on 79 pairs — residual gap = C9. Native suggest path
retired (Ruby pipeline is the ranking authority; native keeps correct?).
Conformance: 2,630 vectors, 0 failures, 0 divergences.

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
