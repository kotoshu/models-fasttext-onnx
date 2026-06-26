#!/usr/bin/env ruby
# frozen_string_literal: true
#
# Walk every models/<lang>/ directory listed in .gitignore's negation
# block (the "active" set), hash the .onnx (and any sibling .vocab.json),
# and emit manifest.json at the repo root. Format matches
# Kotoshu::Integrity::Manifest in the gem.
#
# Re-runnable: scripts/generate_manifest.rb is idempotent modulo the
# generated_at timestamp.
#
# To promote a deferred language: add `!models/<lang>/` to .gitignore,
# `git add models/<lang>/`, re-run this script.

require "digest"
require "json"
require "time"

ROOT = File.expand_path("..", __dir__)
MANIFEST_PATH = File.join(ROOT, "manifest.json")
MODELS_DIR = File.join(ROOT, "models")

# Active languages = those whose .onnx is tracked in git.
# (`git ls-files` honors .gitignore, so deferred dirs stay excluded.)
active_langs = Dir.children(MODELS_DIR).sort.select do |lang|
  lang_dir = File.join(MODELS_DIR, lang)
  File.directory?(lang_dir) && !Dir.glob("#{lang_dir}/*.onnx").empty?
end

resources = {}
languages = []

active_langs.each do |lang|
  lang_dir = File.join(MODELS_DIR, lang)
  tracked = `git ls-files "models/#{lang}/"`.split("\n").map { |p| p.sub("models/#{lang}/", "") }
  next if tracked.empty?
  languages << lang

  meta = {}
  meta_path = File.join(lang_dir, "metadata.json")
  if File.exist?(meta_path)
    meta = JSON.parse(File.read(meta_path, encoding: "UTF-8")) rescue {}
  end

  # OnnxModel.from_file requires <stem>.onnx + <stem>.vocab.json siblings.
  # Only emit entries for tracked files (deferred dirs are .gitignored).
  Dir.children(lang_dir).sort.each do |fname|
    next unless fname.end_with?(".onnx") || fname.end_with?(".vocab.json")
    next unless tracked.include?(fname)
    abs = File.join(lang_dir, fname)
    next unless File.file?(abs)

    bytes = File.read(abs, mode: "rb")
    type = fname.end_with?(".onnx") ? "onnx" : "vocab"

    entry = {
      size: bytes.bytesize,
      sha256: Digest::SHA256.hexdigest(bytes),
      language: lang,
      type: type,
      source: meta["source_model"] || "FastText Common Crawl"
    }
    entry[:fasttext_source] = meta["source_model"] if meta["source_model"]
    entry[:conversion_method] = meta["conversion_method"] if meta["conversion_method"]
    entry[:opset_version] = meta["opset_version"] if meta["opset_version"]
    resources["models/#{lang}/#{fname}"] = entry
  end
end

manifest = {
  version: 1,
  generated_at: Time.now.utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
  repo_version: "v1",
  resource_count: resources.size,
  language_count: languages.size,
  resources: resources
}

File.write(MANIFEST_PATH, JSON.pretty_generate(manifest) + "\n")
puts "Wrote #{MANIFEST_PATH}"
puts "  #{resources.size} resources across #{languages.size} languages"
