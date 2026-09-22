# Plan C9: closing the last ranking gaps to field SymSpell

## Status

pattern 1 (diacritics) executed (2026-09-22): SymSpellStrategy now
indexes folded deletion keys and scores with fold-normalized Damerau
(ä≈a, é≈e, ß≈ss; NFD combining-mark strip). The published de list was
ALSO rebuilt — its ctx-table source was 100% ASCII (umlaut words were
missing from the candidate pool entirely; frequency-list-kelly#5).
de nonword: 70.9 → **72.2%** top-1 (field SymSpell 73.4% = 1 pair on
n=79), top-3 89.9% #1; de realword top-3 15.3 → 18.1%. Patterns 2-4
(vowel substitution, frequency ties, transposition polish) remain
open; at n=79 the residual is inside split noise — wave-2 splits are
the real verdict.

## Problem

We are not sole #1 at top-1 on the German nonword set despite winning
top-3 and top-5 decisively.

## What

For each of the four patterns:
1. implement the smallest change in the gem,
2. extend C1 harness to surface the pattern (error-class breakdown),
3. re-run C1 de + en, freeze reports,
4. publish updated wiki-freq lists if needed.

## Consumers

C1/C8 fleet loop; C6 ranking engine; the "we are #1 across the board"
product claim.
