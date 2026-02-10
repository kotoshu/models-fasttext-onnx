#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative '../tests/test_onnx_model'

##
# Comprehensive test runner for all FastText ONNX models (Ruby version).
#
# Usage:
#   ruby scripts/test_all_models.rb
#

def get_completed_models
  repo_dir = File.expand_path('../..', __FILE__)
  models_dir = File.join(repo_dir, 'models')
  fixtures_dir = File.join(repo_dir, 'tests', 'fixtures')

  return [] unless Dir.exist?(models_dir)

  Dir.glob(File.join(models_dir, '*')).map { |d| File.basename(d) }.select do |lang|
    onnx_file = File.join(models_dir, lang, "fasttext.#{lang}.onnx")
    fixture_file = File.join(fixtures_dir, "#{lang}.yaml")
    File.exist?(onnx_file) && File.exist?(fixture_file)
  end.sort
end

def test_all_models
  completed = get_completed_models

  if completed.empty?
    puts 'No completed models found.'
    return [0, 0, []]
  end

  puts '=' * 80
  puts "TESTING #{completed.size} COMPLETED MODELS (RUBY)"
  puts '=' * 80
  puts

  passed = 0
  failed = 0
  failed_languages = []

  completed.each_with_index do |lang, i|
    puts "[#{i + 1}/#{completed.size}] Testing #{lang.upcase}..."

    tester = ONNXModelTester.new(lang)
    success = tester.run_all_tests

    if success
      passed += 1
    else
      failed += 1
      failed_languages << lang
    end

    puts
  end

  # Summary
  puts '=' * 80
  puts 'TEST SUMMARY'
  puts '=' * 80
  puts "Total: #{completed.size}"
  puts "Passed: #{passed}"
  puts "Failed: #{failed}"

  if failed_languages.any?
    puts
    puts "Failed languages: #{failed_languages.join(', ')}"
  end

  puts
  puts 'Usage:'
  puts '  ruby tests/test_onnx_model.rb <lang>  # Test individual language'
  puts '  ruby scripts/generate_test_fixtures_simple.rb  # Regenerate fixtures'

  [passed, failed, failed_languages]
end

# Main
if __FILE__ == $PROGRAM_NAME
  passed, failed, = test_all_models
  exit(failed.zero? ? 0 : 1)
end
