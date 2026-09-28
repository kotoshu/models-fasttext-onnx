# 5 — Release and announcement

## Release chain (version numbers are OWNER decisions — never guessed)

1. Gem release carrying #235/236/237/238 (owner states the version).
2. kotoshu-server dependency refresh + deployment (TODO.final/3).
3. kotoshu-rs release carrying #58 + the vowelless mirror (owner
   states the version; crates.io semver-forces — check published).
4. @kotoshu/client notes: HTTP parity statement (TODO.final/3).

## Announcement artifact (draft with the frozen table)

- Headline: kotoshu ≥ the strongest open baselines across all 16
  languages at word-level suggestion; per-language margins from the
  frozen table (ar +2.2 / fr +2.8 / vi +2.3 / pl +1.7 / HK +1.7 …).
- The realword story: the fasttext context model's product value —
  ar 26.5 vs 0, de 17.0 vs 0, es 14.0 vs 0 (only lane with signal).
- The honesty story: ar 16.1 → 67.9 came from fixing OUR OWN test
  (303 dictionary-valid pairs masquerading as nonwords) — the number
  changed because the measurement improved, and every step is
  committed.
- Method block: frozen per-class-tagged splits reproducible
  corpus-free, dual-lane field comparison, pinned environment
  (Ruby 3.4.8 / symspellpy 6.10.0 / wordfreq 3.1.1), every claim a
  committed report.
- Surfaces: kotoshu.org models page, gem README, release notes.
- Gate: TODO.final/1 complete; publish = owner review.

## Housekeeping flagged (not acted on)

- scripts/quantize_lane.py untracked in models repo — not ours to
  judge; owner to keep or delete.
- Stale git worktrees in the gem repo (kotoshu-p0, kotoshu-dedup,
  kotoshu-modal-gem*, kotoshu-foldfix, kotoshu-freq4, plan-133
  prunable) — prune on owner say-so.
