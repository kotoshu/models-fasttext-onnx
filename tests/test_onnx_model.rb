#!/usr/bin/env ruby
# frozen_string_literal: true

require 'onnxruntime'
require 'yaml'

##
# Ruby test framework for FastText ONNX models.
#
# Usage:
#   ruby tests/test_onnx_model.rb <language_code>
#

class ONNXModelTester
  def initialize(lang, repo_dir: nil)
    @lang = lang
    @repo_dir = repo_dir || File.expand_path('../..', __FILE__)
    @spec_path = File.join(@repo_dir, 'tests', 'fixtures', "#{lang}.yaml")
    @model_path = File.join(@repo_dir, 'models', lang, "fasttext.#{lang}.onnx")
  end

  def load_spec
    raise "Test spec not found: #{@spec_path}" unless File.exist?(@spec_path)

    YAML.load_file(@spec_path)
  end

  def load_model
    raise "Model not found: #{@model_path}" unless File.exist?(@model_path)

    OnnxRuntime::InferenceSession.new(@model_path)
  end

  def print_model_info(spec, sess)
    metadata = spec['metadata']
    model_specs = spec['model_specifications']

    puts "\n#{metadata['language_name']} (#{metadata['language_code']})"
    puts '-' * 60
    puts "Source: #{metadata['source_model']}"
    puts "Vocabulary: #{metadata['vocab_size'].to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse} words"
    puts "Embedding: #{metadata['embedding_dim']}D"
    puts "ONNX: opset=#{metadata['onnx_opset']}, ir=#{metadata['onnx_ir']}"
    puts

    # Input/output specs
    input_spec = sess.inputs.first
    output_spec = sess.outputs.first
    puts "Input:  #{input_spec[:name]} (#{input_spec[:type]}) #{input_spec[:shape].inspect}"
    puts "Output: #{output_spec[:name]} (#{output_spec[:type]}) #{output_spec[:shape].inspect}"
    puts
  end

  def validate_model_specs(spec, sess)
    model_specs = spec['model_specifications']

    # Check input specs
    input_spec = sess.inputs.first
    input_expected = model_specs['input']

    raise "Input name mismatch: #{input_spec[:name]} != #{input_expected['name']}" unless input_spec[:name] == input_expected['name']
    raise "Input shape mismatch: #{input_spec[:shape].inspect} != #{input_expected['shape'].inspect}" unless input_spec[:shape] == input_expected['shape']

    # Check output specs
    output_spec = sess.outputs.first
    output_expected = model_specs['output']

    raise "Output name mismatch: #{output_spec[:name]} != #{output_expected['name']}" unless output_spec[:name] == output_expected['name']
    raise "Output shape mismatch: #{output_spec[:shape].inspect} != #{output_expected['shape'].inspect}" unless output_spec[:shape] == output_expected['shape']

    puts '  ✓ Model specifications validated'
  end

  def calculate_stats(embedding)
    {
      mean: embedding.sum(0.0) / embedding.size,
      std: Math.sqrt(embedding.map { |v| (v - embedding.sum(0.0) / embedding.size)**2 }.sum(0.0) / embedding.size),
      min: embedding.min,
      max: embedding.max
    }
  end

  def run_test_case(sess, test_case, spec)
    input_spec = sess.inputs.first
    output_spec = sess.outputs.first

    word_index = test_case['input']['word_index']
    puts "\n  Test: #{test_case['name']}"
    puts "    Input: word_index = #{word_index}"

    # Run inference - returns Array of output arrays
    output = sess.run([output_spec[:name]], { input_spec[:name] => [word_index] })
    embedding = output.first

    # Validate output
    expected = test_case['expected_output']
    expected_shape = expected['shape'] || expected['embedding_shape']

    raise "Shape mismatch: #{embedding.size} != #{expected_shape.first}" unless embedding.size == expected_shape.first

    # Check validation rules
    validation = spec['validation_rules'] || {}
    if validation['check_finite']
      nan_values = embedding.select { |v| v.nil? || v.nan? || v.infinite? }
      raise "Contains NaN or Inf" unless nan_values.empty?
    end

    # Print statistics
    stats = calculate_stats(embedding)
    puts "    Output: shape=[#{embedding.size}], mean=#{stats[:mean].round(6)}, std=#{stats[:std].round(6)}"
    puts "            range=[#{stats[:min].round(4)}, #{stats[:max].round(4)}]"

    # Check statistics if provided
    if expected['has_statistics'] && expected['statistics']
      expected_stats = expected['statistics']
      tolerance = 0.000001

      mean_diff = (stats[:mean] - expected_stats['mean']).abs
      std_diff = (stats[:std] - expected_stats['std']).abs

      raise "Mean mismatch: #{stats[:mean]} != #{expected_stats['mean']} ± #{tolerance}" if mean_diff > tolerance
      raise "Std mismatch: #{stats[:std]} != #{expected_stats['std']} ± #{tolerance}" if std_diff > tolerance
    end

    puts '    ✓ Test passed'
    true
  end

  def run_all_tests
    puts '=' * 80
    puts "FASTTEXT ONNX MODEL TEST - #{@lang.upcase}"
    puts '=' * 80

    # Load specification
    spec = load_spec
    puts '✓ Test specification loaded'

    # Load model
    sess = load_model
    puts '✓ ONNX model loaded'

    # Print model info
    print_model_info(spec, sess)

    # Validate model specs
    validate_model_specs(spec, sess)

    # Run test cases
    puts "\nRunning test cases:"
    spec['test_cases'].each do |test_case|
      run_test_case(sess, test_case, spec)
    end

    puts "\n" + '=' * 80
    puts "✓ ALL TESTS PASSED for #{@lang.upcase}"
    puts '=' * 80
    true
  rescue StandardError => e
    puts "\n" + '=' * 80
    puts "✗ TEST FAILED for #{@lang.upcase}"
    puts '=' * 80
    puts "Error: #{e.message}"
    puts e.backtrace.first(10)
    false
  end
end

# Main
if __FILE__ == $PROGRAM_NAME
  if ARGV.size != 1
    puts 'FastText ONNX Model Test Framework (Ruby)'
    puts
    puts 'Usage: ruby tests/test_onnx_model.rb <language_code>'
    puts
    puts 'Examples:'
    puts '  ruby tests/test_onnx_model.rb af'
    puts '  ruby tests/test_onnx_model.rb en'
    puts
    puts 'Test specifications are in tests/fixtures/<lang>.yaml'
    exit 1
  end

  lang = ARGV[0].downcase
  tester = ONNXModelTester.new(lang)
  success = tester.run_all_tests

  exit(success ? 0 : 1)
end
