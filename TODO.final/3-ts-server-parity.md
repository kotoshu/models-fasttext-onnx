# 3 — TS parity: @kotoshu/client

`@kotoshu/client` is an HTTP client; the engine it talks to is
kotoshu-server running the gem. Consequences:

1. **HTTP backend (the published path)**: all gem fixes reach TS users
   the moment kotoshu-server's `gem "kotoshu", "~> 1.0"` resolves to a
   release carrying #235/236/237/238. Action: cut the gem release
   (version = OWNER DECISION), then bump/refresh the server deployment
   and verify /health + a vocalized-Arabic suggest through the public
   API.
2. **WASM backend (unpublished, word-level)**: runs the legacy
   composite — the SymSpell channel is compiled OUT of the wasm feature
   (owner-set 64 MB ceiling; embedded index measured 114.4 MB). The
   wave-2 fixes do NOT reach this path by design. The vowelless fold +
   ingress DO (they live in the shared fold/suggest code, TODO.final/2
   merges into the wasm build automatically). A SymSpell-in-wasm
   redesign (chunked/on-demand tables) is a separate owner-gated arc —
   do not attempt within this program.
3. Client release notes should state: HTTP = full parity (server 1.1.1 carries gem 1.0.8 — the romanization
   channel included); WASM = legacy ranking until the ceiling arc lands.
