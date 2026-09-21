# Plan C7: keyboard-layout universe — IMEs, dual-layout ambiguity

## Status

partially executed (2026-09-21) — gem PR #226: Registry.layouts_for returns
[native, QWERTY] when they differ; EditDistanceStrategy keyboard_penalty takes
min across the set; Chinese IME layouts added (Pinyin/Jyutping/Cangjie/Sucheng)
with regional zh codes; supports_language? exact-match so zh-Hant-TW is not
stolen by bare zh→Pinyin. Remaining: user-config keyboard_layouts: override on
Configuration; IME-specific confusion pairs (tone digits, cangjie radicals);
keyboard-aware error class in C1 harness.

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
