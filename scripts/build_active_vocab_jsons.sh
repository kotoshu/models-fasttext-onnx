#!/usr/bin/env bash
# Download FastText .vec files for the 9 active languages and extract
# .vocab.json siblings for the committed .onnx models.
#
# Idempotent: skips languages whose .vocab.json already exists.
# Caches .vec files under /tmp/fasttext-vec so re-runs don't re-download.
#
# Total download: ~7 GB. Run on a connection without data caps.
#
# Usage:
#   cd /path/to/models-fasttext-onnx
#   ./scripts/build_active_vocab_jsons.sh

set -euo pipefail

ACTIVE_LANGS=(de en es fr pt ru zh ja ko)
VOCAB_SIZE="${VOCAB_SIZE:-100000}"
CACHE_DIR="${CACHE_DIR:-/tmp/fasttext-vec}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$CACHE_DIR"

for lang in "${ACTIVE_LANGS[@]}"; do
  vec_gz="$CACHE_DIR/cc.${lang}.300.vec.gz"
  vec="$CACHE_DIR/cc.${lang}.300.vec"
  vocab="$REPO_ROOT/models/${lang}/fasttext.${lang}.vocab.json"
  onnx="$REPO_ROOT/models/${lang}/fasttext.${lang}.onnx"

  if [[ -f "$vocab" ]]; then
    echo "[skip] $vocab already exists"
    continue
  fi

  if [[ ! -f "$onnx" ]]; then
    echo "[warn] $onnx missing — vocab without .onnx is useless. Skipping."
    continue
  fi

  if [[ ! -f "$vec" ]]; then
    if [[ ! -f "$vec_gz" ]]; then
      echo "[download] cc.${lang}.300.vec.gz"
      curl -fL "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${lang}.300.vec.gz" -o "$vec_gz"
    fi
    echo "[gunzip] $vec_gz"
    gunzip -k "$vec_gz"
  fi

  echo "[extract] $vec -> $vocab"
  python3 "$SCRIPT_DIR/extract_vocab_from_vec.py" "$vec" \
    --vocab-size "$VOCAB_SIZE" \
    --output "$vocab"

  # Reclaim disk: keep only the .gz cache; the .vec can be regenerated
  # if extraction ever needs to run again.
  rm -f "$vec"
done

echo
echo "Done. Regenerate manifest.json with:"
echo "  cd $REPO_ROOT && bundle exec ruby scripts/generate_manifest.rb"
