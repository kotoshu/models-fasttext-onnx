# TODO.final — the closing work program (2026-09-28)

Everything remaining after the wave-2 campaign's engine/data fixes
(all merged: gem #235 #236 #237 #238, kotoshu-rs #58, models #8).
Three API surfaces: **Ruby** (the gem — canonical, all fixes merged),
**Rust** (kotoshu-rs — engine port, dedup + fold-scoped mirror merged),
**TS** (`@kotoshu/client` — HTTP client to kotoshu-server, which runs
the gem; optional unpublished WASM backend running the legacy path).

| # | file | state |
|---|------|-------|
| 1 | final-verdict-freeze | DONE — 16/16 WIN frozen (PR #11); table + generator committed |
| 2 | rust-parity-vowelless | DONE — kotoshu-rs #59 merged |
| 3 | ts-server-parity | UNBLOCKED — gem 1.0.7 on rubygems; server picks up `~> 1.0` on next deploy; wasm ceiling documented |
| 4 | interscript-roadmap | P1/P2/P3 + translit channel — plans with gates, not started |
| 5 | release-and-announcement | DONE — gem 1.0.7 + crates.io 0.3.0 live; site page published (kotoshu.org/models-fasttext-onnx/wave2/); draft finalized |
