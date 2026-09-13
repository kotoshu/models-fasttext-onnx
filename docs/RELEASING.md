# Releasing the models registry

The registry is generated: `scripts/build_registry.py` walks the
descriptors under `models/` and writes `registry.json`. A release cut
publishes the generated assets as a GitHub release and fills the
`primary` URLs with the release-tag convention. This is the runbook.

## The standard cut

1. **Bump the descriptors whose artifacts changed.** Each descriptor
   (`models/<lang>/<artifact>.json`, `models/lid/lid.json`,
   `models/typo/typo.json`) carries its own version and provenance.
2. **Regenerate with the tag** (the tag names the release that will
   carry the assets):

   ```sh
   python3 scripts/build_registry.py --tag vX.Y.Z
   python3 scripts/validate_registry.py
   python3 -m unittest discover -s tests
   ```

   The validator enforces the primary/vocab URL conventions per
   resource family (tiers, LID, packs, typo) and the sha256 pins.
3. **Cut the GitHub release `vX.Y.Z`** with the artifacts the
   regenerated registry references attached (the tier `.onnx` +
   `.vocab.json` pairs, LID, packs, and — once promoted — the typo
   pair).
4. **Commit the regenerated `registry.json`** (release_tag and
   version fields ride it) and push; consumers' `resource_pin` fetch
   it from the default branch.

## The typo bi-encoder promotion (plan 131)

The typo entry ships mirror-only until a release carries its assets.
Promoting it is the descriptor's own `release_tag` knob — the version
choice is the owner's:

1. Set `release_tag` in `models/typo/typo.json` to the tag you are
   about to cut (e.g. `v1.7.0`).
2. Cut the release with BOTH files attached:
   `models/typo/typo.biencoder.onnx` and
   `models/typo/typo.biencoder.vocab.json` (the registry URLs derive
   from these exact names).
3. Regenerate and validate as above — the generator fills `primary`
   and `vocab_url` at the convention automatically; the validator
   demands the pair travel together.

The moment the cut registry is live, the gem side resolves: the
typo layer's `Kotoshu.setup_typo(lang)` / `kotoshu setup LANG --typo`
downloads the pair, and `Engine.for` arms cache-only (plan 131's
two-stage contract; nothing else flips).

## Mirrors

The media-host mirror (`media.githubusercontent.com/media/...` at the
default branch) serves every binary unconditionally — LFS objects
never come from the raw host (pointer stubs). `primary` is the
release URL; the mirror is the offline-tolerant fallback.
