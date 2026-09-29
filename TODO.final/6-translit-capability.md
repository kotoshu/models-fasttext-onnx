# 6 — Transliteration capability class (ar first)

The romanization channel (gem #239) enables a class no field lane can
score: type the romanization, get the native word.

## State

- Gate PASSED locally: 82.5% top-1 / 84.5% top-5 on the 200-pair
  probe (≥ 80% bar). The full 2,000-pair Modal measurement freezes
  the class number.

- Sidecar published: kelly#9 (data/ar.translit.json, 75,382
  fold-normalized ALA-LC keys over 101,307 words).
- Channel merged: gem #239 (ASCII queries on non-Latin languages;
  exact key dist 1, deletion neighborhood dist 2, frequency-ranked).
- Split drafted: eval/realword/ar.suggest3-translit.json (2,000 pairs,
  class "translit", gate-inert by construction — Latin typos cannot
  be dictionary-valid Arabic).
- Capability measurement: bench_modal.py translit_capability()
  (kotoshu lane + field lanes for the record) — running on Modal.

## Gates

- kotoshu top-1 ≥ 80% on the class (the prototype's 200/200 exact-key
  retrieval predicts this).
- Field lanes recorded at ~0 (the honest capability statement).

## After ar

ru (BGN/PCGN), el (ALA-LC 2010), ko (RR) tables from the same builder;
the gem channel is language-agnostic already — only the sidecar is
per-language.
