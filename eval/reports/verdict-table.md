# Wave-2 verdict table (final)

Nonword/realword exact-match top-1 over the frozen dictionary-gated
splits; field = best of the symspell/hunspell lanes on the same
published frequency lists. Gate: kotoshu ≥ every field lane on both
classes. Every number traces to a committed
`eval/reports/suggest-benchmark-{lang}-wave2.json`.

| lang | kotoshu t1 | t3 | t5 | field t1 | rw t1 (k vs field) | gate |
|---|---|---|---|---|---|---|
| en | 87.55 | 96.05 | 98.00 | 86.45 | 2.5 vs 1.0 | WIN |
| de | 87.80 | 95.55 | 97.40 | 86.95 | 17.0 vs 0.0 | WIN |
| es | 85.50 | 96.10 | 97.50 | 84.00 | 14.0 vs 0.0 | WIN |
| fr | 84.25 | 94.80 | 97.25 | 81.45 | 14.0 vs 7.5 | WIN |
| it | 87.20 | 95.90 | 97.80 | 84.30 | 0.0 vs 0.0 | WIN |
| ja | 73.90 | 89.55 | 94.35 | 72.30 | — | WIN |
| ko | 63.13 | 90.45 | 96.55 | 62.93 | 0.0 vs 0.0 | WIN |
| nl | 86.45 | 95.35 | 97.00 | 83.65 | 0.0 vs 0.0 | WIN |
| pl | 88.00 | 96.65 | 98.40 | 86.35 | 0.0 vs 0.0 | WIN |
| pt | 80.15 | 92.85 | 95.00 | 80.95 | 13.0 vs 8.5 | MISS |
| ru | 86.30 | 96.65 | 98.85 | 86.35 | 6.0 vs 0.0 | MISS |
| vi | 68.70 | 82.95 | 88.45 | 66.40 | 0.0 vs 0.0 | WIN |
| zh-Hans-CN | 76.10 | 95.45 | 98.00 | 75.45 | — | WIN |
| zh-Hant-TW | 81.70 | 98.10 | 99.45 | 81.45 | — | WIN |
| zh-Hant-HK | 77.00 | 96.75 | 98.40 | 75.35 | 2.5 vs 0.0 | WIN |
| ar | 67.95 | 88.35 | 94.05 | 65.80 | 26.5 vs 0.0 | WIN |

**14/16 WIN** — 0 pending, 5 stale (de, pt, ru, zh-Hans-CN, zh-Hant-TW)
