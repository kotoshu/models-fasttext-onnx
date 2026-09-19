# Plan D4: browser real-word demo — prove the ecosystem in a browser

## Status

executed (2026-09-19) — demos/realword ships the full page; component parity proven in node against the Python implementations (blake2b hash 2692745670, npz arrays identical to numpy, binary-search lookups identical to bisect: get(5,100)=2118 both sides); scoring validated on fixtures in Python (dual gate: conservative, clean sentences stay clean). The in-browser click-through remains for the owner to open.

## Problem

The new serving architecture (raw-host mirrors for the browser set,
releases for everything else) has never been exercised end to end by an
actual browser workload. The real-word (confusion-set) detection feature
— the reason the models exist beyond dictionary spell check — has no
demonstration. One demo proves both at once.

## What

1. Investigate first (evidence, not assumption): read
   `docs/realword-detection-design.md`, plans 15/16/17, the state of the
   en ctx-LM artifacts (`models/en/fasttext.en.ctx.*`), the en confusion
   set sources, and how the kotoshu.github.io site currently loads models
   (existing wasm/ONNX machinery to reuse; do not disturb in-progress
   site work — add an isolated route).
2. Demo contract: a page where the user types English text; the browser
   fetches `registry.json` + en mini onnx + mini vocab (+ en buckets for
   OOV subword rows where the design calls for them) from raw-host
   mirrors, detects real-word errors from the confusion set (e.g.,
   their/there, lose/loose) by context scoring, and renders the flagged
   words with alternatives. No server, no build-time bundling of models.
3. If a browser-ready en confusion-set artifact does not exist, generate
   a small frozen JSON (top confusion pairs with frequencies) and commit
   it to the plain-git browser set.
4. Gate: the demo must run from the published site (or a static page on
   the site domain) fetching only registry/raw/release URLs, and must
   correctly flag at least the canonical cases in its own fixture.

## Consumers

- The site gains its first semantic-check demo (the "content quality
  checker" story, not just typo correction).
- The demo's languages define `BROWSER_BUCKET_LANGS` growth in D2.
