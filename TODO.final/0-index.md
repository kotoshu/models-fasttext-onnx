# TODO.final — the closing work program (2026-09-28)

Everything remaining after the wave-2 campaign's engine/data fixes
(all merged: gem #235 #236 #237 #238, kotoshu-rs #58, models #8).
Three API surfaces: **Ruby** (the gem — canonical, all fixes merged),
**Rust** (kotoshu-rs — engine port, dedup + fold-scoped mirror merged),
**TS** (`@kotoshu/client` — HTTP client to kotoshu-server, which runs
the gem; optional unpublished WASM backend running the legacy path).

| # | file | state |
|---|------|-------|
| 1 | final-verdict-freeze | wave in flight (14/16 landed); audit + table + docs gated on it |
| 2 | rust-parity-vowelless | DONE this pass (PR merged) |
| 3 | ts-server-parity | server gem bump = release-gated; wasm ceiling documented |
| 4 | interscript-roadmap | P1/P2/P3 + translit channel — plans with gates, not started |
| 5 | release-and-announcement | draft done; gem version + publish = owner decisions |
