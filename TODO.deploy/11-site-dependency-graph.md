# Plan D11: the site's dependency graph — no consumer of the dead pin

## Status

executed (2026-09-20) — CLEAN BILL. (1) kotoshu.github.io declares no @kotoshu dependencies at all: its playground is the server-backed MVP (TODO.impl/63 — the wasm-native path is that repo's documented future target), so the live site never touches the registry pin. (2) @kotoshu/client 0.1.0 has an empty dependencies object — nothing in the client chain pins the worker. (3) kotoshu-js has no @kotoshu model packages. (4) gh-pages realword/ is byte-identical to main's demos/realword (diff -r empty). Verdict: NO consumer of the dead v1.5.0 pin exists anywhere in the ecosystem; the 0.1.1 worker release is hygiene for future installers, not a live rescue. The remaining campaign work lives in TODO.impl (engine arcs: ctx-LM Phase-1 gate per docs/realword-detection-design.md, the Korean subword floor, the ja retrain, fa/lv/pl/sv second sources, gem plan 147) — implementation domain, not deployment.

## Problem

The worker 0.1.1 release fixed the dead v1.5.0 registry pin, but caret
ranges only help fresh installs: a package-lock pins exact versions. If
kotoshu.github.io (the live site) or any package in the ecosystem locks
@kotoshu/worker 0.1.0, its semantic features are broken today. The
demo's gh-pages copy may also have drifted from main's demos/realword.

## What

1. Inspect the lockfiles and manifests of kotoshu.github.io, kotoshu-js,
   and any package depending on @kotoshu/worker or @kotoshu/wasm.
2. Bump pinned dead versions to 0.1.1 where found (each repo via PR or
   its own convention; the site deploy follows its workflow).
3. Verify gh-pages demo parity with main's demos/realword.
4. Freeze findings.

## Consumers

The live site and every npm package in the dependency graph.
