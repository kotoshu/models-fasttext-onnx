# 4 — Interscript roadmap (adopted, gated — not started)

P0 (vowelless normalization) is merged (gem #238). The rest of
interscript's ranked proposal, with OUR acceptance gates:

## P1 — homograph disambiguation for suggestion ranking

Skeleton lookup returns N lemmas (كتب → he wrote / books / was
written); sentence diacritization picks the sense; rank or drop.

- Model: ara-diac-layerdrop-1.0-int4 (95 MB, browser) or the REST
  endpoint (api.interscript.org/v1/infer, open CORS) for the
  zero-download path.
- Gate: build a labeled homograph-context set (≥200 sentences, e.g.
  كتب/كُتُب, عين/عَيْن/عِين senses); adopt only on a measured
  top-1 gain over the undiacritized ranker on that set.
- Tier: S2/S3 context arc — sentence input, not the isolated-word
  bench.

## P2 — vowel-error detection

Where input carries vowels, typed-vs-predicted vocalization
mismatches (harakaat/shadda/i'rab) become a new, explicitly-labeled
error category in check().

- Gate: seeded-error precision (correct vocalized text must NOT
  flag) + recall on the seeded corpus. Depends on P1's diacritizer.

## P3 — phonemic/G2P keys + the romanization channel

- Romanization channel (PROTOTYPED, ours): Interscript.transliterate
  over the kelly lists + fold-normalized keys — 29,382 ar words in
  10.5s, mrhb → مرحبا, 200/200 retrieval. 286 schemes cover
  ara/kor/jpn/ell/rus. Design: offline-generated tables (the
  jyutping/FOLD_TABLE precedent — zero runtime deps), romanization
  keys as a parallel discovery space in the SymSpell strategy.
  Gate: a translit split class (type the romanization, expect the
  native correction) — kotoshu would be the only lane that scores;
  freeze before/with the engine PR.
- fas-g2p / tha-g2p / urd-diac: phonemic candidate generation for
  languages kotoshu does not serve yet — acquisition-arc material,
  not this program.
- Hebrew plene/defective (כתיב מלא/חסר) canonicalization: a
  fold-table extension once he is a served language.

## Standing rule (interscript's, adopted)

Every claimed number reproduces from a public results log or it is
treated as wrong.
