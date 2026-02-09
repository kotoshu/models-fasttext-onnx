#!/usr/bin/env ruby
# frozen_string_literal: true

require 'fileutils'
require 'open3'
require 'json'

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

def test_language(lang)
  model_path = "models/#{lang}/fasttext.#{lang}.onnx"

  unless File.exist?(model_path)
    return { lang => { status: 'missing', model_path: model_path } }
  end

  script = <<~PYTHON
    import sys
    import numpy as np
    try:
      import onnxruntime as ort
      sess = ort.InferenceSession('#{model_path}', providers=['CPUExecutionProvider'])
      input_name = sess.get_inputs()[0].name
      output_name = sess.get_outputs()[0].name

      # Test multiple indices
      for idx in [0, 1, 10, 100, 1000, 10000]:
        emb = sess.run([output_name], {input_name: np.array([idx], dtype=np.int64)})[0]
        assert emb.shape == (300,), f"Wrong shape: {emb.shape}"
        assert np.all(np.isfinite(emb)), "Contains NaN or Inf"

      mean = np.mean(sess.run([output_name], {input_name: np.array([0], dtype=np.int64)})[0])
      print(f"OK:{mean:.6f}")
      sys.exit(0)
    except Exception as e:
      print(f"FAIL:{e}")
      sys.exit(1)
  PYTHON

  stdout, stderr, status = Open3.capture3('python3', '-c', script)

  if status.success?
    mean = stdout.strip.split(':').last.to_f
    { lang => { status: 'ok', mean: mean, size: File.size(model_path) } }
  else
    { lang => { status: 'failed', error: stderr.strip } }
  end
end

def main
  puts '=' * 80
  puts 'COMPREHENSIVE ONNX MODEL VERIFICATION'
  puts '=' * 80
  puts

  results = {}
  success = 0
  failed = 0
  missing = 0

  ALL_LANGUAGES.each_with_index do |lang, idx|
    print "[#{idx + 1}/#{ALL_LANGUAGES.size}] #{lang.upcase.ljust(6)} ... "

    result = test_language(lang)
    results.merge!(result)

    lang_result = result[lang]
    case lang_result[:status]
    when 'ok'
      size_mb = (lang_result[:size] / (1024.0 * 1024.0)).round(2)
      puts "✓ OK (#{size_mb} MB, mean=#{lang_result[:mean].round(6)})"
      success += 1
    when 'missing'
      puts "✗ MISSING"
      missing += 1
    when 'failed'
      puts "✗ FAILED: #{lang_result[:error]}"
      failed += 1
    end
  end

  puts
  puts '=' * 80
  puts 'VERIFICATION SUMMARY'
  puts '=' * 80
  puts "Total languages: #{ALL_LANGUAGES.size}"
  puts "✓ Passed: #{success}"
  puts "✗ Failed: #{failed}"
  puts "Missing: #{missing}"
  puts

  if failed + missing == 0
    puts '✓ ALL MODELS VERIFIED SUCCESSFULLY'
    puts
    puts 'All 157 FastText ONNX models are functional and ready for use!'
    exit 0
  else
    puts '✗ SOME MODELS FAILED VERIFICATION'
    puts
    puts 'Failed languages:'
    results.each do |lang, data|
      puts "  - #{lang}: #{data[:status]}" unless data[:status] == 'ok'
    end
    exit 1
  end
end

main if __FILE__ == $PROGRAM_NAME
