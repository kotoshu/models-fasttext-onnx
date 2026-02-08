#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"
require "json"
require "digest"

# Comprehensive ONNX verification script

class ONNXVerifier
  SUPPORTED_LANGUAGES = %w[de en es fr pt ru].freeze

  def verify_all
    puts "=" * 80
    puts "COMPREHENSIVE ONNX VERIFICATION"
    puts "=" * 80
    puts

    results = {}

    SUPPORTED_LANGUAGES.each do |lang|
      results[lang] = verify_language(lang)
    end

    print_summary(results)
  end

  def verify_language(lang)
    puts "Verifying #{lang.upcase}..."
    puts "-" * 40

    result = {
      language: lang,
      checks: [],
      status: "PASS"
    }

    # Check 1: ONNX file exists
    onnx_file = "deployment/onnx/#{lang}/fasttext.#{lang}.onnx"
    if File.exist?(onnx_file)
      size = File.size(onnx_file)
      result[:checks] << { name: "ONNX file exists", status: "PASS", size_mb: (size.to_f / 1024 / 1024).round(2) }
    else
      result[:checks] << { name: "ONNX file exists", status: "FAIL" }
      result[:status] = "FAIL"
      return result
    end

    # Check 2: Metadata exists
    metadata_file = "deployment/onnx/#{lang}/metadata.json"
    if File.exist?(metadata_file)
      result[:checks] << { name: "Metadata file exists", status: "PASS" }
      metadata = JSON.parse(File.read(metadata_file))

      # Verify metadata fields
      required_fields = %w[language type file checksum cached_at conversion_method]
      required_fields.each do |field|
        if metadata[field]
          result[:checks] << { name: "Metadata has #{field}", status: "PASS", value: metadata[field] }
        else
          result[:checks] << { name: "Metadata has #{field}", status: "FAIL" }
          result[:status] = "FAIL"
        end
      end

      # Check language matches
      if metadata["language"] == lang
        result[:checks] << { name: "Metadata language matches", status: "PASS" }
      else
        result[:checks] << { name: "Metadata language matches", status: "FAIL" }
        result[:status] = "FAIL"
      end

      # Check type is 'onnx'
      if metadata["type"] == "onnx"
        result[:checks] << { name: "Metadata type is 'onnx'", status: "PASS" }
      else
        result[:checks] << { name: "Metadata type is 'onnx'", status: "FAIL" }
        result[:status] = "FAIL"
      end
    else
      result[:checks] << { name: "Metadata file exists", status: "FAIL" }
      result[:status] = "FAIL"
    end

    # Check 3: SHA256 checksum exists
    checksum_file = "deployment/onnx/#{lang}/SHA256"
    if File.exist?(checksum_file)
      stored_checksum = File.read(checksum_file).strip
      calculated_checksum = Digest::SHA256.file(onnx_file).hexdigest

      if stored_checksum == calculated_checksum
        result[:checks] << { name: "SHA256 checksum valid", status: "PASS" }
      else
        result[:checks] << { name: "SHA256 checksum valid", status: "FAIL",
                             stored: stored_checksum, calculated: calculated_checksum }
        result[:status] = "FAIL"
      end
    else
      result[:checks] << { name: "SHA256 checksum exists", status: "FAIL" }
      result[:status] = "FAIL"
    end

    # Check 4: Python ONNX validation
    python_valid = validate_with_python(onnx_file, lang)
    if python_valid
      result[:checks] << { name: "Python ONNX validation", status: "PASS",
                           vocab_size: python_valid[:vocab_size],
                           embed_dim: python_valid[:embed_dim] }
    else
      result[:checks] << { name: "Python ONNX validation", status: "FAIL" }
      result[:status] = "FAIL"
    end

    # Check 5: File size is reasonable (100-500 MB)
    size_mb = File.size(onnx_file).to_f / 1024 / 1024
    if size_mb >= 100 && size_mb <= 500
      result[:checks] << { name: "File size reasonable", status: "PASS", size_mb: size_mb.round(2) }
    else
      result[:checks] << { name: "File size reasonable", status: "FAIL", size_mb: size_mb.round(2) }
      result[:status] = "FAIL"
    end

    result
  end

  def validate_with_python(onnx_path, lang)
    script = <<~PYTHON
      import sys
      try:
        import onnx
        import numpy as np

        model = onnx.load('#{onnx_path}')

        # Check graph exists
        assert model.graph is not None, "Graph is None"

        # Check input/output
        assert len(model.graph.input) > 0, "No inputs"
        assert len(model.graph.output) > 0, "No outputs"

        # Get input/output details
        input_name = model.graph.input[0].name
        output_name = model.graph.output[0].name

        # Check metadata
        meta = {p.key: p.value for p in model.metadata_props}

        vocab_size = int(meta.get('vocabulary_size', 0))
        embed_dim = int(meta.get('embedding_dimension', 0))

        # Verify graph structure
        assert len(model.graph.node) > 0, "No nodes in graph"

        # Check for Constant and Gather nodes
        node_types = [n.op_type for n in model.graph.node]
        assert 'Constant' in node_types, "No Constant node"
        assert 'Gather' in node_types, "No Gather node"

        print(f"vocab_size={vocab_size},embed_dim={embed_dim},nodes={len(model.graph.node)}")
        sys.exit(0)
      except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    PYTHON

    require "open3"
    stdout, stderr, status = Open3.capture3("python3", "-c", script)

    if status.success?
      parts = stdout.strip.split(",")
      Hash[parts.map { |p| p.split("=").map(&:to_sym) }]
    else
      nil
    end
  end

  def print_summary(results)
    puts
    puts "=" * 80
    puts "VERIFICATION SUMMARY"
    puts "=" * 80
    puts

    all_pass = true

    SUPPORTED_LANGUAGES.each do |lang|
      result = results[lang]
      status = result[:status] == "PASS" ? "✓ PASS" : "✗ FAIL"

      size_mb = result[:checks].find { |c| c[:size_mb] }&.dig(:size_mb)
      vocab_size = result[:checks].find { |c| c[:vocab_size] }&.dig(:vocab_size)
      embed_dim = result[:checks].find { |c| c[:embed_dim] }&.dig(:embed_dim)

      puts "#{lang.upcase}: #{status}"
      puts "  Size: #{size_mb} MB" if size_mb
      puts "  Vocabulary: #{vocab_size} words" if vocab_size
      puts "  Embedding: #{embed_dim}D" if embed_dim

      if result[:status] != "PASS"
        failed_checks = result[:checks].select { |c| c[:status] == "FAIL" }
        if failed_checks.any?
          puts "  Failed checks:"
          failed_checks.each do |check|
            puts "    - #{check[:name]}"
          end
        end
      end

      puts

      all_pass = false if result[:status] != "PASS"
    end

    puts "=" * 80
    if all_pass
      puts "ALL VERIFICATIONS PASSED ✓"
    else
      puts "SOME VERIFICATIONS FAILED ✗"
    end
    puts "=" * 80
  end
end

if __FILE__ == $PROGRAM_NAME
  verifier = ONNXVerifier.new
  verifier.verify_all
end
