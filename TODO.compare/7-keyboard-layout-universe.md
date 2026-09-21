# Plan C7: keyboard-layout universe — IMEs, dual-layout ambiguity

## Status

scoped (2026-09-21). Gem ships 17 layouts (arabic, azerty, bulgarian, devanagari_inscript, dubeolsik, dvorak, greek, hebrew, hrsl, jcuken, latin, persian, qwerty, qwertz, serbian, turkish_q, ukrainian). Registry.layout_for picks ONE layout per language_code (exact → base → script default → QWERTY). Missing: Chinese IME layouts (pinyin≈QWERTY keys with tone digits, cangjie, sucheng, cantonese/jyutping). Owner constraint: a user may type on a **native** layout OR plain **QWERTY** (or any Latin layout) — we can only guess, accept config, or detect via declarative signal; we might never know.

## Problem

Typos are keyboard-dependent. Ranking that assumes the wrong layout mis-scores proximity penalties. CJK input is not a single layout: pinyin/cangjie/sucheng/jyutping are IME methods over QWERTY (or Cangjie-key) physical keys. Arabic, Korean (dubeolsik), Devanagari already have layouts but must be verified wired through language_code. Dual-layout ambiguity is structural — one layout_for is not enough when the user might be on either.

## What

1. **Inventory + wiring audit.** For every supported language_code, record: default layout, alternate layouts users actually use, whether Registry.layout_for returns the right default today. Fix any miss (e.g. de→QWERTZ, fr→AZERTY, ko→dubeolsik, ar→arabic).
2. **Chinese IME layouts.** Add pinyin, cangjie, sucheng, cantonese/jyutping as Layout classes. Physical key topology for pinyin/jyutping is QWERTY; the "typo" space is Latin-letter + tone-digit / final confusion, not CJK glyph proximity. Cangjie/sucheng have their own radical-key maps.
3. **Dual-layout scoring.** Scorer accepts a layout set (primary + alternates), not a single layout. keyboard_penalty = min(penalty under each layout in the set). Default set per language: [native_default, qwerty] for languages whose native layout ≠ QWERTY; [qwerty] otherwise. User config / declarative override: `keyboard_layout:` or `keyboard_layouts:` on Configuration and per-call.
4. **Detection (optional, later).** If surrounding text or OS locale is known, prefer that layout; never hard-fail when unknown — fall back to the dual set.
5. **Eval.** Extend C1 harness with a keyboard-aware error class (key-adjacent substitutions generated from each layout) so layout wiring is measured, not assumed.

## Consumers

EditDistanceStrategy / KeyboardProximityStrategy keyboard_penalty; C6 ranking; every non-English language's top-1.
