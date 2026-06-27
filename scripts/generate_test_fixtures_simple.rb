#!/usr/bin/env ruby
# frozen_string_literal: true

require 'yaml'
require 'json'
require 'fileutils'
require 'open3'

##
# Generate test fixture YAML files for completed FastText ONNX models.
#
# For each completed model, creates tests/fixtures/<lang>.yaml with:
# - Metadata (vocab size, embedding dim, source)
# - Model specifications (input/output)
# - Validation rules
# - Test cases with actual embedding samples
#

# Only generate for completed models
COMPLETED_MODELS = Dir['models/*'].select { |d| File.directory?(d) && File.exist?(File.join(d, "fasttext.#{File.basename(d)}.onnx")) }.map { |d| File.basename(d) }

SCRIPT_DIR = File.dirname(__FILE__)
SAMPLE_SCRIPT = File.join(SCRIPT_DIR, 'sample_embedding.py')
VOCAB_SCRIPT = File.join(SCRIPT_DIR, 'get_vocab_size.py')

def get_vocab_size(model_path)
  # Get actual vocab size from ONNX model
  output, = Open3.capture3('python3', VOCAB_SCRIPT, model_path)
  return nil if output.strip.empty?

  output.strip.to_i
rescue => e
  puts "    ⊗ Warning: Could not get vocab size: #{e.message}"
  nil
end

def get_sample_embedding(model_path, index)
  # Use Python helper script to sample embedding
  output, = Open3.capture3('python3', SAMPLE_SCRIPT, model_path, index.to_s)

  # Parse JSON output
  return nil if output.strip.empty? || output.include?('ERROR')

  JSON.parse(output.strip)
rescue JSON::ParserError => e
  puts "    ⊗ Warning: Could not parse JSON for index #{index}: #{e.message}"
  nil
end

def generate_fixture_for_language(lang)
  model_path = File.join(SCRIPT_DIR, '..', 'models', lang, "fasttext.#{lang}.onnx")

  puts "  Generating #{lang.upcase}..."

  # Get actual vocab size from ONNX model
  vocab_size = get_vocab_size(model_path) || 100_000

  # Sample embeddings at different indices (ensure indices are valid)
  samples = {}
  test_indices = [0, 1]
  test_indices << [vocab_size / 2, vocab_size - 1].min if vocab_size > 2

  test_indices.each do |idx|
    next if idx >= vocab_size  # Skip if index is out of bounds

    result = get_sample_embedding(model_path, idx)

    if result
      samples[idx] = {
        'statistics' => result['stats'],
        'sample_values' => result['samples']
      }
    else
      puts "    ⊗ Warning: Could not sample index #{idx}"
    end
  end

  # Create fixture
  fixture = {
    'metadata' => {
      'language_code' => lang,
      'language_name' => lang.upcase,
      'source_model' => "cc.#{lang}.300.vec",
      'vocab_size' => vocab_size,
      'embedding_dim' => 300,
      'onnx_opset' => 11,
      'onnx_ir' => 11
    },
    'model_specifications' => {
      'input' => {
        'name' => 'word_index',
        'type' => 'int64',
        'shape' => [1],
        'description' => 'Single word index from vocabulary'
      },
      'output' => {
        'name' => 'embedding',
        'type' => 'float32',
        'shape' => [300],
        'description' => '300-dimensional word embedding'
      }
    },
    'validation_rules' => {
      'check_finite' => true,
      'check_shape' => [300],
      'embedding_dim' => 300
    },
    'test_cases' => [
      {
        'name' => 'first_word',
        'input' => { 'word_index' => 0 },
        'expected_output' => {
          'shape' => [300],
          'has_statistics' => samples.key?(0),
          'statistics' => samples.dig(0, 'statistics') || {}
        }
      }
    ]
  }

  # Add second word test if vocab size allows
  if vocab_size > 1
    fixture['test_cases'] << {
      'name' => 'second_word',
      'input' => { 'word_index' => 1 },
      'expected_output' => {
        'shape' => [300],
        'has_statistics' => samples.key?(1),
        'statistics' => samples.dig(1, 'statistics') || {}
      }
    }
  end

  # Add middle word test if vocab size allows
  if vocab_size > 2
    fixture['test_cases'] << {
      'name' => 'middle_word',
      'input' => { 'word_index' => vocab_size / 2 },
      'expected_output' => {
        'shape' => [300]
      }
    }
  end

  # Add last word test
  fixture['test_cases'] << {
    'name' => 'last_word',
    'input' => { 'word_index' => vocab_size - 1 },
    'expected_output' => {
      'shape' => [300]
    }
  }

  # Write fixture
  fixtures_dir = File.join(SCRIPT_DIR, '..', 'tests', 'fixtures')
  FileUtils.mkdir_p(fixtures_dir)

  fixture_path = File.join(fixtures_dir, "#{lang}.yaml")
  File.write(fixture_path, fixture.to_yaml)

  { lang => { status: 'generated' } }
end

def main
  puts '=' * 80
  puts 'GENERATING TEST FIXTURES FOR COMPLETED MODELS'
  puts '=' * 80
  puts

  puts "Found #{COMPLETED_MODELS.size} completed models"
  puts

  results = {}
  generated = 0

  COMPLETED_MODELS.each do |lang|
    result = generate_fixture_for_language(lang)
    results.merge!(result)

    if result[lang][:status] == 'generated'
      generated += 1
    end
  end

  puts
  puts '=' * 80
  puts "FIXTURE GENERATION COMPLETE"
  puts '=' * 80
  puts "Generated: #{generated}"
  puts
  puts "Test fixtures created in: tests/fixtures/"
  puts
  puts "Usage:"
  puts "  python3 tests/test_onnx_model.py <lang>"
  puts "  Example: python3 tests/test_onnx_model.py af"
  puts
  puts "Note: Run this again after all 157 models are converted."
end

main if __FILE__ == $PROGRAM_NAME
