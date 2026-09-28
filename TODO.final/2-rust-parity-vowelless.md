# 2 — Rust parity: vowelless normalization (interscript P0 mirror)

The gem strips Arabic haraqat + Hebrew niqqud at the Generator ingress
and in fold_word (Suggestions::VOWELLESS_MARKS, gem #238). kotoshu-rs
must mirror both for API-contract parity — the rs engine serves
whatever dictionary it is handed, and en-only data does not excuse a
divergent contract (same reasoning as the #5908ce4 fold-scoping
mirror).

## Steps

1. `symspell.rs fold_word`: chars in U+0591-05BD/05BF/05C1-05C2/
   05C4-05C5/05C7/064B-065F/0670 are DROPPED (the gem returns [] for
   them — they are not fold units at all).
2. `suggest/mod.rs suggest()`: strip the same ranges from `word` at
   ingress, before the `dictionary.correct` check.
3. Tests: vocalized Arabic folds equal to plain; a vocalized typo
   produces the unvocalized slate; ASCII unaffected ("drinks ate"
   keeps its space — the gem's own regression).
4. PR → CI → rebase-merge; the wasm build inherits it (the legacy
   composite path also uses fold_word).

## Note

Escapes-only single-line regex literals on the Ruby side exist because
a multiline /x class once swallowed a literal newline+space; the Rust
range check is char-compare based and immune, but keep the same
one-line discipline.
