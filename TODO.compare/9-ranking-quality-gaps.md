# Plan C9: closing the last ranking gaps to field SymSpell

## Status

open (2026-09-21). After C6 (frequency SymSpell + composite ranked:true):
kotoshu de nonword top-5 91.1% / top-3 86.1% / top-1 70.9% vs field
SymSpell 73.4% (all slices) and Hunspell 54.4%. Top-1 gap of ~3pp on
79 pairs (small sample, ±6pp CI). The 25 top-1 misses cluster into
four mechanical patterns — each a one-design-change fix:

1. **Umlaut/extended-letter handling.** SymSpell deletes against a
   single-byte view of the word; substitutions like ö→ö (same letter,
   different encoding) or ß→ss miss the index. Example: `gejöscht` →
   `gelöscht` not found; `fäälig` → `fällig` not found. Fix: normalize
   word + dictionary to NFC + fold common equivalents before indexing
   (and reverse at output).

2. **Vowel substitution.** `sue` → `sie` (i/u substitution, deletion
   index misses). Edit-distance distance-1 includes substitutions but
   SymSpell's deletion-neighborhood only generates single deletions.
   Fix: add an EditDistance pre-pass at distance-1 to fill substitutions
   when SymSpell's pool is sparse.

3. **Frequency-tie ordering.** Cases where truth shares distance 1
   with a higher-frequency false positive (`benutzte` vs `benutze`,
   `bereits` vs `beits`). Fix: small per-char keyboard-proximity
   penalty inside SymSpell's tiebreak — the true correction's
   substitution pattern tends to involve adjacent keys more often
   than the random miss.

4. **Transpositions not handled by SymSpell by default.** Several
   typos are letter-swaps. SymSpellStrategy has a handle_transpositions
   flag — verify it is on (default true per source) and that the
   per-language precompute reuses it.

Each fix is small, has a clear eval signal, and folds into C6's
frequency SymSpell channel.

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
