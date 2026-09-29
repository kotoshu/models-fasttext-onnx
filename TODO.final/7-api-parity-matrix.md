# 7 — API parity matrix (Ruby / Rust / TS)

| capability | Ruby gem | Rust (kotoshu-rs) | TS (@kotoshu/client) |
|---|---|---|---|
| variant-pure zh lists | ✅ #236 | n/a (en-embedded; documented) | ✅ via server (gem 1.0.8+) |
| ranked tail dedup | ✅ #237 | ✅ #58 | ✅ HTTP via server; WASM = legacy path (64 MB ceiling, owner-set) |
| vowelless normalization | ✅ #238 | ✅ #59 | ✅ HTTP via server; WASM inherits the fold |
| romanization channel | ✅ #239 + 1.0.8 | rs serves en only — documented constraint | ✅ HTTP via server (1.0.8+); WASM: no |
| vocalized-input lookup | ✅ | rs correct? is native-dictionary-driven (tolerant) | ✅ via server |

The RS parity constraints are structural (the en-only embedded index;
the wasm memory ceiling) and recorded in TODO.final/3 — not defects.
The TS HTTP path is always at gem parity one server deploy behind a
release.
