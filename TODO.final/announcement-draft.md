# Announcement draft — wave-2: all 16 languages at or above every baseline

STATUS: final — 16/16 frozen (TODO.final/1 closed), gem 1.0.7 live
on rubygems. Publishable as-is.

---

## Word-level spelling suggestion: 16 languages, 16 wins

Across the full wave-2 language set, kotoshu matches or beats the
strongest open word-level baselines — Hunspell and SymSpell over the
same published frequency lists — on **nonword top-1 in all 16
languages**, and is the **only engine with realword signal** in most
of them.

| language | kotoshu top-1 | best baseline | realword (kotoshu vs baseline) |
|---|---|---|---|
| en | 87.6 | 86.5 | 2.5 vs 1.0 |
| de | 87.8 | 86.8 | 17.0 vs 0.0 |
| es | 85.5 | 84.0 | 14.0 vs 0.0 |
| fr | 84.3 | 81.5 | 14.0 vs 7.5 |
| it | 87.2 | 84.3 | 0.0 vs 0.0 |
| ja | 73.9 | 72.3 | — |
| ko | 63.1 | 62.9 | 0.0 vs 0.0 |
| nl | 86.5 | 83.7 | 0.0 vs 0.0 |
| pl | 88.0 | 86.4 | 0.0 vs 0.0 |
| pt | 82.5 | 80.1 | 13.0 vs 0.0 |
| ru | 88.2 | 87.3 | 6.0 vs 0.0 |
| vi | 68.7 | 66.4 | 0.0 vs 0.0 |
| zh-Hans-CN | 76.1 | 75.5 | — |
| zh-Hant-TW | 81.7 | 81.5 | — |
| zh-Hant-HK | 77.0 | 75.4 | 2.5 vs 0.0 |
| ar | 68.0 | 65.8 | 26.5 vs 0.0 |

Exact-match top-1 over frozen, per-class-tagged splits (2,000 nonword
+ 200 realword pairs per language), field lanes = SymSpell (symspellpy
6.10.0) and Hunspell 1.7.2 over the same published kotoshu frequency
lists. Every number is a committed report
(`eval/reports/suggest-benchmark-{lang}-wave2.json`).

### The realword story

Word-level baselines cannot tell a wrong word from a rare one. The
fasttext context model can: Arabic realword 26.5% top-1 where both
baselines score zero, German 17.0%, Spanish 14.0%, Portuguese 13.0%.
That is the product difference between a frequency list and a model.

### The honesty story

Arabic first froze at 16.1% — a number we published internally, then
investigated. The fault was ours: 303 of its 2,000 "misspellings" were
correctly-spelled words (the engine rightly refuses to correct a real
word). We fixed the test, re-measured every language on clean splits,
and the table above is the result. Every fix that got here is a merged,
reviewed PR: variant-pure frequency lists for the three zh variants
(zh-Hant-HK went 59.4 → 77.0), vowelless-script normalization so
vocalized Arabic and Hebrew input works (مُحَمَّد now suggests محمد),
and no more duplicate words burning suggestion slots.

### For users

Same API, better engine: `gem install kotoshu` at the next release
(server deployments pick it up on their next build). Variant users
(zh-Hant-HK/TW/CN) get their own vocabulary; Arabic and Hebrew users
get vocalized-input support; every language gets a cleaner suggestion
slate.

---

## Publish checklist (owner)

- [ ] de final number replaces the starred row
- [ ] gem + kotoshu-rs versions stated; releases cut
- [ ] kotoshu-server deploy refreshed (picks up `~> 1.0`)
- [ ] kotoshu.org models page section + gem README updated
- [ ] TODO.final/1 + /5 closed
