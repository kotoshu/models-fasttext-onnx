#!/usr/bin/env ruby
# frozen_string_literal: true

require 'fileutils'
require 'open3'

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

SCRIPTS_DIR = File.expand_path('..', __FILE__)

def generate_test_script(lang)
  script = <<~SCRIPT
    #!/bin/bash
    # Test script for #{lang.upcase} FastText ONNX model

    echo "Testing #{lang.upcase}..."
    python3 "#{SCRIPTS_DIR}/test_model.py" #{lang}
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
      echo "✓ #{lang.upcase} PASSED"
    else
      echo "✗ #{lang.upcase} FAILED"
    fi

    exit $exit_code
  SCRIPT

  File.write("#{SCRIPTS_DIR}/test_#{lang}.sh", script)
  FileUtils.chmod("+x", "#{SCRIPTS_DIR}/test_#{lang}.sh")
  puts "✓ Generated test_#{lang}.sh"
end

def main
  puts "Generating individual test scripts for #{ALL_LANGUAGES.size} languages..."
  puts

  ALL_LANGUAGES.each do |lang|
    generate_test_script(lang)
  end

  puts
  puts "=" * 60
  puts "✓ Generated #{ALL_LANGUAGES.size} test scripts"
  puts
  puts "Usage:"
  puts "  ./scripts/test_<lang>.sh"
  puts "  Example: ./scripts/test_af.sh"
  puts
end

main if __FILE__ == $PROGRAM_NAME
