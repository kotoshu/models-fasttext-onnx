# D8 sweep evidence (2026-09-19)

Scope: all 12 kotoshu repositories (rb/py/js/ts/rs/go/toml/json/yml/adoc/md).

## Dead model URLs found

1. kotoshu (gem) — `lib/kotoshu/source_registry.rb` + `spec/kotoshu/source_registry_spec.rb:35`
   - `MEDIA_BASE_URL = "https://media.githubusercontent.com/media/kotoshu"` builds
     `.onnx` model URLs on the LFS media host. The host 404s everything (LFS
     store empty by design since TODO.deploy/1).
   - Impact: NONE live today - `ModelRegistry` entries cover all 57 languages,
     so the legacy path only fires as a fallback; the spec is a pure
     URL-construction assertion (no network).
   - Recommended fix (gem-side, own plan/PR): drop the media-host branch;
     mini-tier files are plain git (raw host serves them); full-tier files
     are release assets (fetch via the registry primary, which the newer
     ModelCache path already does). Update the spec to the raw-host URL.
   - Not touched here: the gem has in-flight work (plan 147) - collision risk.

## Clean

kotoshu-py, kotoshu-js, kotoshu-rs, kotoshu-go, kotoshu-vscode, kotoshu-lsp,
kotoshu-server, kotoshu.github.io, action-kotoshu, dictionaries,
frequency-list-kelly: zero dead model URLs (registry-driven).

## Gem chain verified against the new architecture

- Configured registry base (configuration.rb:71) =
  github.com/kotoshu/models-fasttext-onnx/raw/main -> serves the v1.8.0
  registry (232 resources, raw mirrors, null release-only mirrors, zh retired).
