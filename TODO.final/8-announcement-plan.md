# 8 — Announcement plan: how we tell users

The achievement, in one sentence: **kotoshu beats the strongest open
word-level baselines in all 16 languages and ships a cross-script
capability none of them have — reproducibly, client-side, with every
number a committed report.**

## The audience ladder (tell the technical truth at each rung)

1. **Existing users (the shipped surfaces)** — they discover the wins
   by upgrading; the announcement explains what changed.
   - gem README (the rubygems landing page): results table + the
     new-in-1.0.8 cross-script retrieval + the honesty story.
   - kotoshu.org wave2 page (LIVE): the full matrix.
   - kotoshu-server 1.1.0 / kotoshu-rs 0.3.0 release notes: parity
     statements.
2. **Ruby community** — where the gem lives.
   - RubyWeekly / Ruby Flow / ruby-talk submission: lead with "16
     languages, 16 wins over Hunspell and SymSpell, client-side" + the
     honesty story (it is the differentiator — every benchmark page
     claims wins; almost none publish their own failed first number
     and the fix).
   - A Show HN draft ("Show HN: Client-side spelling that beats
     Hunspell and SymSpell in 16 languages") — owner posts, draft
     ready; the comment section will ask exactly the method questions
     the method block already answers.
3. **The i18n/NLP niche** — the real differentiator audience.
   - The variant-pure zh story (HK/TW/CN each with their own
     vocabulary) and the vocalized-Arabic fix speak to maintainers of
     ICU/LibreOffice-style dictionaries; the romanization channel
     (type mrhb, get مرحبا) speaks to transliteration communities
     (the interlibt interlibtangle: Interscript itself).
   - Cross-post where the user's own networks live; drafts ready.
4. **Social (owner's voice)** — 280-char and thread-length drafts:
   the matrix screenshot + "every number is a committed report" + the
   ar 16.1→68.0 honesty beat.

## Sequence

| step | surface | owner action |
|---|---|---|
| 1 | gem README + site (done/live) | review |
| 2 | release notes (1.0.7/1.0.8, rs 0.3.0, server 1.1.0) | review |
| 3 | RubyWeekly/ruby-talk submission | approve + send |
| 4 | Show HN draft | owner posts |
| 5 | social drafts | owner posts |

## Rules

- Every number traces to a committed report; the de row refresh note
  and the split-gate story are told as part of the win, not hidden.
- No competitor disparagement: the baselines are named respectfully
  (they are the standards we measure against; SymSpell and Hunspell
  are magnificent — and we beat them at their own game).
- Language claims match served scope: 16 languages measured; the gem
  serves 57 models — only the 16 make benchmark claims.
