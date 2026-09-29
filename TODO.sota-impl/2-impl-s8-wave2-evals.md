# Impl S8: wave-2 realistic evals (graph node S8)

## Status

DONE (2026-09-29). 16/16 languages meet the gate — kotoshu ≥ every
field lane on nonword AND realword — on the final engine (gem
#235/#236/#237/#238) and dictionary-gated splits (models #8). The
verdict table is frozen in eval/reports/verdict-table.md (generator:
scripts/final_verdict_table.py); every number is a committed report.
Releases live: gem 1.0.7 (rubygems), kotoshu-rs 0.3.0 (crates.io),
kotoshu-server 1.1.0 (kotoshu 1.0.7). Results page:
kotoshu.org/models-fasttext-onnx/wave2/. S8-E4 extension (zh-Hant-HK/
ko/vi/ar) closed under TODO.sota-impl/11; the closing program is
TODO.final. En route: the split dictionary gate (ar 303/it 44/vi 30/
ko 26 dictionary-valid pairs removed — the ar 16.1 anomaly's true
mechanism), tail dedup in gem+rs, vi raw scoring, variant-pure lists,
vowelless normalization (interscript P0).

## Gate

G-S8: all languages frozen ≥2,000 nonword; wave-1 verdicts re-earned or revised HONESTLY.

## Consumers

The re-earned "#1 across the board" claim; S2/S3/S9 gates.
