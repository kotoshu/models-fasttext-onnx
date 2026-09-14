# Plan 10: Mirror-URL fetchability validation

## Status: executed

## Problem

The registry's `urls.mirror` convention assumes the artifact is LFS-tracked
(the media host `media.githubusercontent.com` only serves LFS objects). Plan
136's KTM1 matrix landed as a **plain git blob** because `.gitattributes` had
no `*.ktm1` rule, so the live mirror URL 404'd — `kotoshu setup en --typo`
could not download the matrix. `validate_registry.py` verified descriptor
sha/size and URL *construction*, but never that the URL actually **serves**.
PR CI was green; the bug surfaced only in a live-client check.

## Fix (shipped separately)

- PR #40: `*.ktm1` LFS rule + renormalization (content sha unchanged, no
  registry regeneration needed).

## This plan — close the validation gap

1. `validate_registry.py --check-urls`: probe every non-null `urls.primary`,
   `urls.mirror`, and `vocab_url` with a ranged GET (`Range: bytes=0-0`,
   30s timeout, one retry on 5xx/network errors; 4xx is immediately fatal).
   200/206 = reachable. When the resource declares `size_bytes`, the
   `Content-Range` total must match — catches truncated/renamed uploads.
   Vocab sidecars are reachability-only (their size lives in per-tier ground
   truth, not the entry). 8 workers; deterministic (sorted) error order.
2. Hermetic tests (`tests/test_registry.py`): a real local `http.server`
   serves a temp fixture; good registry passes, a 404 mirror fails, a size
   mismatch fails. No internet.
3. Wire `--check-urls` into CI (every PR) and the release gate
   (`release.yml`, alongside `--check-files`) — a broken mirror must never
   reach a tag.

## Why ranged GET, not HEAD

The media host's HEAD behavior is inconsistent for LFS redirects; a 1-byte
GET confirms real bytes flow and yields the authoritative total size.

## Evidence

- `python3 scripts/validate_registry.py --check-urls` against the live
  registry on main: all non-null URLs probe OK **after** PR #40; before it,
  exactly the ktm1 mirror fails — reproducing the incident.
- `python3 -m unittest tests.test_registry -v` green.
