#!/usr/bin/env ruby
# frozen_string_literal: true

require 'open3'
require 'fileutils'

##
# Verify all ONNX models in the repository.
#
# Tests each language model:
# 1. Loads with onnxruntime
# 2. Returns 300D embeddings
# 3. Handles multiple queries
# 4. Has valid floating-point values
#

SUPPORTED_LANGUAGES = %w[de en es fr pt ru].freeze

def test_language(lang)
  puts "Testing #{lang.upcase}..."

  script = <<~PYTHON
    import sys
    import numpy as np

    try:
      import onnx
      import onnxruntime as ort

      # Load model
      model = onnx.load('models/#{lang}/fasttext.#{lang}.onnx')
      print(f"  IR version: {model.ir_version}, Opset: {model.opset_import[0].version}")

      # Create inference session
      sess = ort.InferenceSession('models/#{lang}/fasttext.#{lang}.onnx', providers=['CPUExecutionProvider'])

      # Get input/output names
      input_name = sess.get_inputs()[0].name
      output_name = sess.get_outputs()[0].name

      # Test multiple indices
      for idx in [0, 1, 100, 1000]:
        embedding = sess.run([output_name], {input_name: np.array([idx], dtype=np.int64)})[0][0]

        # Verify
        assert embedding.shape == (300,), f"Index {idx} has wrong shape: {embedding.shape}"
        assert np.all(np.isfinite(embedding)), f"Index {idx} contains NaN or Inf"

      # Calculate stats for index 0
      embedding = sess.run([output_name], {input_name: np.array([0], dtype=np.int64)})[0][0]
      mean = np.mean(embedding)
      std = np.std(embedding)
      min_val = np.min(embedding)
      max_val = np.max(embedding)

      print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
      print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
      print(f"  ✓ PASS")
      sys.exit(0)

    except Exception as e:
      print(f"  ✗ FAIL: {e}")
      sys.exit(1)
  PYTHON

  stdout, stderr, status = Open3.capture3('python3', '-c', script)

  if status.success?
    puts stdout
    true
  else
    puts stderr
    false
  end
end

def verify_all_models
  puts '=' * 80
  puts 'ONNX MODEL VERIFICATION'
  puts '=' * 80
  puts

  # Check if models directory exists
  unless Dir.exist?('models')
    puts 'Error: models directory not found'
    puts 'Run this script from the repository root'
    return false
  end

  results = {}
  all_pass = true

  SUPPORTED_LANGUAGES.each do |lang|
    model_path = "models/#{lang}/fasttext.#{lang}.onnx"
    unless File.exist?(model_path)
      puts "✗ #{lang.upcase}: Model file not found: #{model_path}"
      all_pass = false
      next
    end

    results[lang] = test_language(lang)
    all_pass = false unless results[lang]
    puts
  end

  puts '=' * 80
  if all_pass
    puts 'ALL ONNX MODELS VERIFIED ✓'
    puts
    puts 'Summary:'
    SUPPORTED_LANGUAGES.each do |lang|
      puts "  #{lang.upcase}: ✓ PASS"
    end
    puts
    puts 'All models are functional and ready for use!'
  else
    puts 'SOME MODELS FAILED ✗'
    puts
    puts 'Summary:'
    SUPPORTED_LANGUAGES.each do |lang|
      status = results[lang] ? '✓ PASS' : '✗ FAIL'
      puts "  #{lang.upcase}: #{status}"
    end
  end
  puts '=' * 80

  all_pass
end

# Run verification
if __FILE__ == $PROGRAM_NAME
  success = verify_all_models
  exit(success ? 0 : 1)
end
