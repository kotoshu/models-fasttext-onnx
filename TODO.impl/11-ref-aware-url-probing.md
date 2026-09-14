# Plan 11: Ref-aware mirror probing (--urls-ref)

## Status: executed

## Problem

Plan 10's `--check-urls` gate probes the registry's mirror URLs as
written — `…/main/<path>`. An artifact **added on a PR branch** does not
exist on `main` until the merge, so the gate 404s on exactly the entries
the PR adds: any PR shipping a new registry artifact (a new language's
KTM1 matrix, a new tier) can never pass its own fetchability check. The
gate is self-blocking for the very case it was built to protect.

## Fix

- `--urls-ref REF` (default `main`): `mirror_probe_url` rewrites the
  `MEDIA_URL/main/` prefix to `MEDIA_URL/REF/` before probing. LFS
  objects are content-addressed and served at every ref carrying them
  (verified live: all 599 URLs probe OK at both `main` and a commit
  sha), so the branch's copy is byte-identical to what lands on merge.
  Error messages keep the registry's canonical URL.
- CI and the release gate pass `--urls-ref "${{ github.ref_name }}"` —
  identity on main/tag runs, branch-correct on PRs.
- Tests: pure-function rewrite cases (main identity, None, non-mirror
  release URLs untouched) + a hermetic full-stack test where the local
  HTTP server answers only under the branch path, proving the probe
  followed the ref and not the `/main/` the registry names.

## Evidence

- `python3 -m unittest discover -s tests -p "test_registry.py"` —
  18 tests OK.
- Live: `--check-urls --urls-ref main` and `--urls-ref af1fd99` both
  report `URLs OK: 599 URL(s) reachable`.

## Unblocks

Plan 12 (KTM1 matrices for de/es/fr/pt/ru) — its PR adds five
mirror-only artifacts that only exist at the branch ref.
