# Plan 19: BCP-47 registry language codes (the variant-model prerequisite)

## Status: executed (models-repo half) - the gem half is plan 147

Owner approval 2026-09-17: "code, yes" - the registry accepts BCP-47
variant codes so zh can split into zh-Hans-CN / zh-Hant-TW / zh-Hant-HK
(the regions differ by vocabulary, not just script; see
docs/cjk-typo-corpus-research.md).

## What changed

- schemas/registry.schema.json: the model-entry language pattern, the
  pack-entry language pattern, and the resource-id pattern all widen
  from `[a-z]{2}` to
  `[a-z]{2,3}(-[A-Z][a-z]{3})?(-([A-Z]{2}|[0-9]{3}))?` - language +
  optional script subtag (capitalized: Hans/Hant) + optional region
  (uppercase two-letter or three-digit). CANONICAL CASING IS ENFORCED:
  zh-hans-cn is invalid, exactly one spelling per variant ships.
  Three-letter ISO bases (the old nds rejection class) now pass too.
- tests/test_registry.py: construction-level tests - a fixture whose
  model entry carries each approved variant validates green (mirror
  URL + manifest entry rewritten per convention), and a
  lowercase-subtag fixture is REJECTED. Suite 65/65.

Deliberately NOT changed: no variant entries exist yet (they arrive
with the variant models, plan 18a); the legacy `zh` entry stays until
the replacement cut; build_registry.py needs no change (it emits what
the manifest describes).

## Consumers

The gem half is plan 147 (proposed): accept variant codes through the
registry reader, cache ids ("{lang}:onnx..." with subtags), setup CLI,
and the resolver; pack keys follow the same widening. The zh-Hans-CN
model build (plan 18a execution) is the first entry consumer.
