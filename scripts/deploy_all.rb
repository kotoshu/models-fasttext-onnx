#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative "lib/kotoshu/cache/model_cache"
require_relative "lib/kotoshu/version"
require "fileutils"
require "json"
require "time"

# Comprehensive ONNX conversion, testing, and deployment script
# This script handles all supported languages: de, en, es, fr, pt, ru

class ONNXManager
  SUPPORTED_LANGUAGES = %w[de en es fr pt ru].freeze
  EXPECTED_VOCAB_SIZES = {
    "de" => 1_000_000,
    "en" => 2_000_000,
    "es" => 1_000_000,
    "fr" => 1_000_000,
    "pt" => 1_000_000,
    "ru" => 1_000_000
  }.freeze

  attr_reader :cache, :results

  def initialize
    @cache = Kotoshu::Cache::ModelCache.new
    @results = {}
  end

  def run_all
    puts "=" * 80
    puts "COMPLETE ONNX CONVERSION, TESTING, AND DEPLOYMENT"
    puts "=" * 80
    puts
    puts "Supported languages: #{SUPPORTED_LANGUAGES.join(', ')}"
    puts "Expected processing time: ~2-4 hours (depending on network speed)"
    puts
    puts "Steps:"
    puts "  1. Verify FastText models"
    puts "  2. Convert to ONNX (with progress tracking)"
    puts "  3. Verify ONNX content (vocabulary, dimensions, checksum)"
    puts "  4. Test local caching"
    puts "  5. Prepare for GitHub deployment"
    puts
    print "Continue? (y/n) "
    return unless gets.chomp.downcase == 'y'

    SUPPORTED_LANGUAGES.each do |lang|
      process_language(lang)
    end

    print_summary
  end

  def process_language(lang)
    puts
    puts "=" * 80
    puts "PROCESSING: #{lang.upcase}"
    puts "=" * 80

    result = {
      language: lang,
      fasttext: nil,
      onnx: nil,
      tests: [],
      errors: []
    }

    # Step 1: Get FastText model
    puts "\n[1/5] Downloading/verifying FastText model..."
    result[:fasttext] = get_fasttext_model(lang)

    # Step 2: Convert to ONNX
    puts "\n[2/5] Converting to ONNX format..."
    result[:onnx] = convert_to_onnx(lang)

    # Step 3: Verify ONNX content
    puts "\n[3/5] Verifying ONNX content..."
    verify_onnx_content(lang, result)

    # Step 4: Test caching
    puts "\n[4/5] Testing local cache..."
    test_local_cache(lang, result)

    # Step 5: Prepare for deployment
    puts "\n[5/5] Preparing for GitHub deployment..."
    prepare_for_deployment(lang, result)

    @results[lang] = result

    print_language_summary(result)
  end

  def get_fasttext_model(lang)
    model_path = @cache.get_fasttext_model(lang)

    unless model_path && File.exist?(model_path)
      { error: "FastText model not found for #{lang}" }
      return
    end

    size_mb = File.size(model_path).to_f / 1024 / 1024
    line_count = `wc -l #{model_path.shellescape}`.to_i

    {
      path: model_path,
      size_mb: size_mb.round(2),
      line_count: line_count,
      status: "OK"
    }
  rescue StandardError => e
    { error: e.message }
  end

  def convert_to_onnx(lang)
    # Clear ONNX cache to force fresh conversion
    onnx_cache_dir = File.expand_path("~/.kotoshu/models/#{lang}/models/onnx")
    FileUtils.rm_rf(onnx_cache_dir) if Dir.exist?(onnx_cache_dir)

    start_time = Time.now

    model_path = @cache.get_onnx_model(lang)

    elapsed = Time.now - start_time

    unless model_path && File.exist?(model_path)
      return { error: "ONNX conversion failed for #{lang}" }
    end

    size_mb = File.size(model_path).to_f / 1024 / 1024

    {
      path: model_path,
      size_mb: size_mb.round(2),
      conversion_time: elapsed.round(2),
      status: "OK"
    }
  rescue StandardError => e
    { error: e.message }
  end

  def verify_onnx_content(lang, result)
    onnx_path = result.dig(:onnx, :path)
    return unless onnx_path

    tests = []

    # Test 1: File exists and is readable
    tests << { name: "File exists", status: File.exist?(onnx_path) ? "PASS" : "FAIL" }

    # Test 2: File size is reasonable (100-500 MB)
    size_mb = File.size(onnx_path).to_f / 1024 / 1024
    size_ok = size_mb >= 100 && size_mb <= 500
    tests << { name: "File size reasonable (#{size_mb.round(2)} MB)", status: size_ok ? "PASS" : "FAIL" }

    # Test 3: ONNX file format
    format_ok = verify_onnx_format(onnx_path)
    tests << { name: "ONNX format valid", status: format_ok ? "PASS" : "FAIL" }

    # Test 4: Metadata exists
    metadata_path = File.dirname(onnx_path) + "/metadata.json"
    metadata_ok = File.exist?(metadata_path)
    tests << { name: "Metadata exists", status: metadata_ok ? "PASS" : "FAIL" }

    if metadata_ok
      metadata = JSON.parse(File.read(metadata_path))
      tests << { name: "Metadata language matches", status: metadata['language'] == lang ? "PASS" : "FAIL" }
      tests << { name: "Metadata type is 'onnx'", status: metadata['type'] == 'onnx' ? "PASS" : "FAIL" }
      tests << { name: "Metadata has checksum", status: metadata['checksum'] ? "PASS" : "FAIL" }
      tests << { name: "Metadata has conversion_method", status: metadata['conversion_method'] ? "PASS" : "FAIL" }
    end

    # Test 5: Python ONNX validation
    python_validation = validate_with_python(onnx_path, lang)
    tests << { name: "Python ONNX validation", status: python_validation ? "PASS" : "FAIL",
               details: python_validation }

    result[:tests] = tests
  end

  def verify_onnx_format(onnx_path)
    # Read first 4 bytes - should be ONNX magic number
    File.open(onnx_path, 'rb') do |f|
      magic = f.read(4)
      # ONNX files start with 'ONNX' in binary
      magic == "\x08\x05\x00\x00" || magic.start_with?("ONNX")
    end
  rescue StandardError
    false
  end

  def validate_with_python(onnx_path, lang)
    script = <<~PYTHON
      import sys
      try:
        import onnx
        model = onnx.load('#{onnx_path}')
        # Check graph exists
        assert model.graph is not None
        # Check input/output
        assert len(model.graph.input) > 0
        assert len(model.graph.output) > 0
        # Check metadata
        meta = {p.key: p.value for p in model.metadata_props}
        assert 'vocabulary_size' in meta
        assert 'embedding_dimension' in meta
        print(f"OK: vocab_size={meta['vocabulary_size']}, embed_dim={meta['embedding_dimension']}")
        sys.exit(0)
      except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    PYTHON

    require 'open3'
    stdout, stderr, status = Open3.capture3('python3', '-c', script)

    if status.success?
      puts "    ✓ #{stdout.strip}"
      true
    else
      puts "    ✗ #{stderr.strip}"
      false
    end
  end

  def test_local_cache(lang, result)
    tests = []

    # Test 1: Cache hit on second access
    onnx_path1 = @cache.get_onnx_model(lang)
    onnx_path2 = @cache.get_onnx_model(lang)
    tests << { name: "Cache returns same path", status: onnx_path1 == onnx_path2 ? "PASS" : "FAIL" }

    # Test 2: Cached ONNX works
    onnx_path = result.dig(:onnx, :path)
    tests << { name: "ONNX file accessible", status: onnx_path && File.exist?(onnx_path) ? "PASS" : "FAIL" }

    result[:tests].concat(tests)
  end

  def prepare_for_deployment(lang, result)
    onnx_path = result.dig(:onnx, :path)
    return unless onnx_path

    deployment_dir = File.expand_path("deployment/onnx/#{lang}")
    FileUtils.mkdir_p(deployment_dir)

    # Copy ONNX file
    dest_onnx = File.join(deployment_dir, File.basename(onnx_path))
    FileUtils.cp(onnx_path, dest_onnx)

    # Copy and update metadata
    metadata_path = File.dirname(onnx_path) + "/metadata.json"
    dest_metadata = File.join(deployment_dir, "metadata.json")

    if File.exist?(metadata_path)
      metadata = JSON.parse(File.read(metadata_path))
      # Add deployment info
      metadata["deployed_at"] = Time.now.utc.iso8601
      metadata["kotoshu_version"] = Kotoshu::VERSION
      File.write(dest_metadata, JSON.pretty_generate(metadata))
    end

    # Create checksum file
    require 'digest'
    checksum = Digest::SHA256.file(onnx_path).hexdigest
    File.write(File.join(deployment_dir, "SHA256"), checksum)

    result[:deployment] = {
      directory: deployment_dir,
      files: Dir.glob("#{deployment_dir}/*").map { |f| File.basename(f) },
      status: "READY"
    }
  end

  def print_language_summary(result)
    puts
    puts "Summary for #{result[:language].upcase}:"
    puts "-" * 40

    if result[:fasttext]
      puts "  FastText: #{result[:fasttext][:status]}"
      puts "    Size: #{result[:fasttext][:size_mb]} MB"
      puts "    Lines: #{result[:fasttext][:line_count]}"
    end

    if result[:onnx]
      puts "  ONNX: #{result[:onnx][:status]}"
      puts "    Size: #{result[:onnx][:size_mb]} MB"
      puts "    Time: #{result[:onnx][:conversion_time]}s"
    end

    passed = result[:tests].count { |t| t[:status] == "PASS" }
    total = result[:tests].size
    puts "  Tests: #{passed}/#{total} passed"

    if result[:deployment]
      puts "  Deployment: #{result[:deployment][:status]}"
      puts "    Directory: #{result[:deployment][:directory]}"
    end

    if result[:errors].any?
      puts "  Errors:"
      result[:errors].each { |e| puts "    - #{e}" }
    end
  end

  def print_summary
    puts
    puts "=" * 80
    puts "OVERALL SUMMARY"
    puts "=" * 80

    SUPPORTED_LANGUAGES.each do |lang|
      result = @results[lang]
      next unless result

      status = if result[:errors].any?
                 "FAILED"
               elsif result[:tests].all? { |t| t[:status] == "PASS" }
                 "SUCCESS"
               else
                 "PARTIAL"
               end

      puts "#{lang.upcase}: #{status}"
    end

    puts
    puts "Next steps:"
    puts "  1. Review deployment/ directory"
    puts "  2. Run: cd deployment/onnx && ../deploy_to_github.sh"
    puts "  3. Update ModelCache to download from GitHub"
    puts "  4. Test remote caching with cleared cache"
  end
end

# String extension for shell escaping
class String
  def shellescape
    gsub(/([^A-Za-z0-9_\-.,:\/@\n])/n, '\\\\\\1').gsub(/\n/, "'\n'")
  end
end

if __FILE__ == $PROGRAM_NAME
  manager = ONNXManager.new
  manager.run_all
end
