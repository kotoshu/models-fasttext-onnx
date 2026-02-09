#!/usr/bin/env ruby
# frozen_string_literal: true

require 'fileutils'
require 'open3'
require 'json'
require 'digest'

BASE_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl'
VOCAB_SIZE = 100_000
SCRIPT_DIR = File.dirname(__FILE__)
REPO_DIR = File.expand_path('..', SCRIPT_DIR)
MODELS_DIR = File.join(REPO_DIR, 'models')
DOWNLOADS_DIR = File.join(REPO_DIR, 'downloads')
CONVERSION_SCRIPT = File.join(SCRIPT_DIR, 'fasttext_to_onnx.py')

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

def process_language(lang)
  puts "\n[#{lang}] Processing..."

  vec_gz = File.join(DOWNLOADS_DIR, "cc.#{lang}.300.vec.gz")
  vec_file = File.join(DOWNLOADS_DIR, "cc.#{lang}.300.vec")
  lang_dir = File.join(MODELS_DIR, lang)
  onnx_file = File.join(lang_dir, "fasttext.#{lang}.onnx")

  # Skip if already done
  if File.exist?(onnx_file) && File.size(onnx_file) > 100_000
    puts "  ✓ Already converted"
    return true
  end

  FileUtils.mkdir_p(lang_dir)

  # Download if needed
  unless File.exist?(vec_file)
    puts "  Downloading..."
    url = "#{BASE_URL}/cc.#{lang}.300.vec.gz"
    system("wget -q -O '#{vec_gz}' '#{url}'") || system("curl -sL -o '#{vec_gz}' '#{url}'")
    system("gunzip -c '#{vec_gz}' > '#{vec_file}'")
  end

  # Convert
  puts "  Converting #{VOCAB_SIZE} words..."
  system("python3 '#{CONVERSION_SCRIPT}' '#{vec_file}' '#{onnx_file}' --vocab-size #{VOCAB_SIZE} > /dev/null")

  # Verify
  puts "  Verifying..."
  result = system("python3 -c \"import onnxruntime as ort; import numpy as np; s=ort.InferenceSession('#{onnx_file}'); e=s.run(None, {s.get_inputs()[0].name: np.array([0], dtype=np.int64)})[0]; assert e.shape==(300,); exit(0)\"")

  if result
    puts "  ✓ SUCCESS"
    FileUtils.rm_f(vec_file)  # Clean up to save space
    FileUtils.rm_f(vec_gz)
    return true
  else
    puts "  ✗ FAILED"
    return false
  end
end

def main
  puts "Converting #{ALL_LANGUAGES.size} FastText languages to ONNX..."
  puts "Vocab size: #{VOCAB_SIZE}, Output dir: #{MODELS_DIR}"
  puts

  FileUtils.mkdir_p(DOWNLOADS_DIR)
  FileUtils.mkdir_p(MODELS_DIR)

  success = 0
  failed = 0

  ALL_LANGUAGES.each_with_index do |lang, i|
    if process_language(lang)
      success += 1
    else
      failed += 1
    end

    puts "Progress: #{i + 1}/#{ALL_LANGUAGES.size} (#{success} success, #{failed} failed)"
  end

  puts "\n✓ Complete: #{success} converted, #{failed} failed"
end

main
