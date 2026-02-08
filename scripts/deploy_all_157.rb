#!/usr/bin/env ruby
# frozen_string_literal: true

require 'fileutils'
require 'open3'
require 'json'
require 'digest'

##
# Download and convert ALL 157 FastText language models to ONNX format.
#
# This script will:
# 1. Download all FastText .vec.gz files (4-5 GB each)
# 2. Convert each to ONNX format (114 MB each with 100K vocab)
# 3. Verify each model works
# 4. Generate metadata
#
# Estimated time: 20-40 hours depending on bandwidth
# Estimated storage: ~700 GB temporary (downloads) + ~18 GB final (ONNX)
#

BASE_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl'
VOCAB_SIZE = 100_000
SCRIPT_DIR = File.expand_path('..', __dir__)
REPO_DIR = File.dirname(SCRIPT_DIR)
MODELS_DIR = File.join(REPO_DIR, 'models')
DOWNLOAD_DIR = File.join(SCRIPT_DIR, 'downloads')
CONVERSION_SCRIPT = File.join(SCRIPT_DIR, 'fasttext_to_onnx.py')

# All 157 FastText language codes
ALL_LANGUAGES = %w[
  af als am an ar arz as ast az azb ba bar bcl be bg bh bn bo bpy br bs
  ca ce ceb ckb co cs cv cy da de diq dv el eml en eo es et eu fa fi frr
  fy ga gd gl gom gu gv he hi hif hr hsb ht hu hy ia id ilo io is it ja jv
  ka kk km kn ko ku ky la lb li lmo lt lv mai mg mhr min mk ml mn mr mrj
  ms mt mwl my myv mzn nah nap nds ne new nl nn no nso oc or os pa pam
  pfl pl pms pnb ps pt qu rm ro ru sa sah sc scn sco sd sh si sk sl so
  sq sr su sv sw ta te tg th tk tl tr tt ug uk ur uz vec vi vls vo wa
  war xmf yi yo zea zh
].freeze

def ensure_directories
  FileUtils.mkdir_p(DOWNLOAD_DIR)
  FileUtils.mkdir_p(MODELS_DIR)
  puts "✓ Directories ready"
end

def download_language(lang)
  vec_file = File.join(DOWNLOAD_DIR, "cc.#{lang}.300.vec")
  vec_gz = "#{vec_file}.gz"
  url = "#{BASE_URL}/cc.#{lang}.300.vec.gz"

  if File.exist?(vec_file)
    puts "  ✓ Already downloaded: #{lang}"
    return vec_file
  end

  puts "  Downloading #{lang} (4-5 GB)..."
  start_time = Time.now

  # Download with wget or curl
  if system('which wget > /dev/null 2>&1')
    system("wget -O '#{vec_gz}' '#{url}' 2>&1 | tail -10")
  elsif system('which curl > /dev/null 2>&1')
    system("curl -L -o '#{vec_gz}' '#{url}' 2>&1 | tail -10")
  else
    raise "Neither wget nor curl found. Please install one."
  end

  # Decompress
  puts "  Decompressing..."
  system("gunzip -c '#{vec_gz}' > '#{vec_file}'")
  FileUtils.rm_f(vec_gz)

  elapsed = Time.now - start_time
  size_mb = File.size(vec_file) / (1024.0 * 1024.0)
  puts "  ✓ Downloaded in #{elapsed.floor / 3600}h#{(elapsed % 3600).floor / 60}m (#{size_mb.floor} MB)"

  vec_file
end

def convert_to_onnx(lang, vec_file)
  lang_dir = File.join(MODELS_DIR, lang)
  FileUtils.mkdir_p(lang_dir)

  onnx_file = File.join(lang_dir, "fasttext.#{lang}.onnx")

  if File.exist?(onnx_file)
    puts "  ✓ Already converted: #{lang}"
    return onnx_file
  end

  puts "  Converting #{lang} to ONNX..."
  start_time = Time.now

  # Run conversion
  cmd = "python3 '#{CONVERSION_SCRIPT}' '#{vec_file}' '#{onnx_file}' --vocab-size #{VOCAB_SIZE}"
  output, = Open3.capture3(cmd)

  elapsed = Time.now - start_time
  size_mb = File.size(onnx_file) / (1024.0 * 1024.0)
  puts "  ✓ Converted in #{elapsed.floor}s (#{size_mb.floor} MB)"

  onnx_file
end

def verify_model(lang, onnx_file)
  puts "  Verifying #{lang}..."

  script = <<~PYTHON
    import sys
    import onnxruntime as ort
    import numpy as np

    try:
      sess = ort.InferenceSession('#{onnx_file}', providers=['CPUExecutionProvider'])
      input_name = sess.get_inputs()[0].name
      output_name = sess.get_outputs()[0].name

      # Test multiple indices
      for idx in [0, 1, 100]:
        emb = sess.run([output_name], {input_name: np.array([idx], dtype=np.int64)})[0][0]
        assert emb.shape == (300,), f"Wrong shape: {emb.shape}"
        assert np.all(np.isfinite(emb)), "Contains NaN or Inf"

      print("  ✓ PASS")
      sys.exit(0)
    except Exception as e:
      print(f"  ✗ FAIL: {e}")
      sys.exit(1)
  PYTHON

  _, _, status = Open3.capture3('python3', '-c', script)
  status.success?
end

def generate_metadata(lang, onnx_file)
  metadata = {
    version: Time.now.utc.iso8601,
    language: lang,
    type: 'onnx',
    file: "fasttext.#{lang}.onnx",
    checksum: Digest::SHA256.file(onnx_file).hexdigest,
    source_model: "cc.#{lang}.300.vec",
    conversion_method: 'fasttext_to_onnx.py',
    opset_version: 11,
    vocab_size: VOCAB_SIZE,
    embedding_dim: 300
  }

  metadata_file = File.join(File.dirname(onnx_file), 'metadata.json')
  File.write(metadata_file, JSON.pretty_generate(metadata))

  metadata_file
end

def process_language(lang)
  puts "Processing #{lang.upcase}..."

  begin
    # Download
    vec_file = download_language(lang)

    # Convert
    onnx_file = convert_to_onnx(lang, vec_file)

    # Verify
    verify_model(lang, onnx_file)

    # Generate metadata
    generate_metadata(lang, onnx_file)

    puts "✓ #{lang.upcase} completed"
    true
  rescue StandardError => e
    puts "✗ #{lang.upcase} failed: #{e.message}"
    false
  end
end

def main
  puts "=" * 80
  puts "FASTTEXT TO ONNX - ALL 157 LANGUAGES"
  puts "=" * 80
  puts
  puts "This will download and convert all 157 FastText language models."
  puts "Estimated time: 20-40 hours"
  puts "Estimated storage: ~700 GB temporary + ~18 GB final"
  puts
  print "Continue? [y/N] "
  return unless $stdin.gets.chomp =~ /^[Yy]/

  puts
  puts "Starting conversion..."
  puts

  ensure_directories

  results = {}
  success_count = 0
  failure_count = 0

  ALL_LANGUAGES.each_with_index do |lang, idx|
    puts "\n[#{idx + 1}/#{ALL_LANGUAGES.size}] #{lang.upcase}"
    puts '-' * 40

    if process_language(lang)
      success_count += 1
    else
      failure_count += 1
    end

    # Cleanup source file to save space
    vec_file = File.join(DOWNLOAD_DIR, "cc.#{lang}.300.vec")
    FileUtils.rm_f(vec_file) if File.exist?(vec_file)
  end

  puts
  puts "=" * 80
  puts "CONVERSION COMPLETE"
  puts "=" * 80
  puts "Success: #{success_count}"
  puts "Failed: #{failure_count}"
  puts
  puts "Total ONNX models: #{ALL_LANGUAGES.size}"
  puts "Total size: ~#{ALL_LANGUAGES.size * 114} MB (~#{(ALL_LANGUAGES.size * 114) / 1024} GB)"
  puts
end

if __FILE__ == $PROGRAM_NAME
  main
end
