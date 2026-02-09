#!/usr/bin/env ruby
# frozen_string_literal: true

require 'yaml'
require 'json'
require 'fileutils'
require 'open3'

##
# Generate test fixture YAML files for each FastText ONNX language.
#
# For each language, creates tests/fixtures/<lang>.yaml with:
# - Metadata (vocab size, embedding dim, source)
# - Model specifications (input/output)
# - Validation rules
# - Test cases (first word, middle word, last word)
# - Sample statistics with actual embeddings
#

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

LANGUAGE_NAMES = {
  'af' => 'Afrikaans', 'als' => 'Alemannic', 'am' => 'Amharic', 'an' => 'Aragonese',
  'ar' => 'Arabic', 'arz' => 'Egyptian Arabic', 'as' => 'Assamese', 'ast' => 'Asturian',
  'az' => 'Azerbaijani', 'azb' => 'Southern Azerbaijani', 'ba' => 'Bashkir',
  'bar' => 'Bavarian', 'bcl' => 'Central Bicolano', 'be' => 'Belarusian',
  'bg' => 'Bulgarian', 'bh' => 'Bihari', 'bn' => 'Bengali', 'bo' => 'Tibetan',
  'bpy' => 'Bishnupriya Manipuri', 'br' => 'Breton', 'bs' => 'Bosnian',
  'ca' => 'Catalan', 'ce' => 'Chechen', 'ceb' => 'Cebuano', 'ckb' => 'Kurdish (Sorani)',
  'co' => 'Corsican', 'cs' => 'Czech', 'cv' => 'Chuvash', 'cy' => 'Welsh',
  'da' => 'Danish', 'de' => 'German', 'diq' => 'Zazaki', 'dv' => 'Dhivehi',
  'el' => 'Greek', 'eml' => 'Emilian-Romagnol', 'en' => 'English', 'eo' => 'Esperanto',
  'es' => 'Spanish', 'et' => 'Estonian', 'eu' => 'Basque', 'fa' => 'Persian',
  'fi' => 'Finnish', 'fr' => 'French', 'frr' => 'North Frisian', 'fy' => 'West Frisian',
  'ga' => 'Irish', 'gd' => 'Scottish Gaelic', 'gl' => 'Galician', 'gom' => 'Goan Konkani',
  'gu' => 'Gujarati', 'gv' => 'Manx', 'he' => 'Hebrew', 'hi' => 'Hindi',
  'hif' => 'Fiji Hindi', 'hr' => 'Croatian', 'hsb' => 'Upper Sorbian', 'ht' => 'Haitian',
  'hu' => 'Hungarian', 'hy' => 'Armenian', 'ia' => 'Interlingua', 'id' => 'Indonesian',
  'ilo' => 'Ilokano', 'io' => 'Ido', 'is' => 'Icelandic', 'it' => 'Italian',
  'ja' => 'Japanese', 'jv' => 'Javanese', 'ka' => 'Georgian', 'kk' => 'Kazakh',
  'km' => 'Khmer', 'kn' => 'Kannada', 'ko' => 'Korean', 'ku' => 'Kurdish (Kurmanji)',
  'ky' => 'Kirghiz', 'la' => 'Latin', 'lb' => 'Luxembourgish', 'li' => 'Limburgish',
  'lmo' => 'Lombard', 'lt' => 'Lithuanian', 'lv' => 'Latvian', 'mai' => 'Maithili',
  'mg' => 'Malagasy', 'mhr' => 'Meadow Mari', 'min' => 'Minangkabau', 'mk' => 'Macedonian',
  'ml' => 'Malayalam', 'mn' => 'Mongolian', 'mr' => 'Marathi', 'mrj' => 'Hill Mari',
  'ms' => 'Malay', 'mt' => 'Maltese', 'mwl' => 'Mirandese', 'my' => 'Burmese',
  'myv' => 'Erzya', 'mzn' => 'Mazandarani', 'nah' => 'Nahuatl', 'nap' => 'Neapolitan',
  'nds' => 'Low Saxon', 'ne' => 'Nepali', 'new' => 'Newar', 'nl' => 'Dutch',
  'nn' => 'Norwegian (Nynorsk)', 'no' => 'Norwegian (Bokmål)', 'nso' => 'Northern Sotho',
  'oc' => 'Occitan', 'or' => 'Oriya', 'os' => 'Ossetian', 'pa' => 'Eastern Punjabi',
  'pam' => 'Kapampangan', 'pfl' => 'Palatinate German', 'pl' => 'Polish', 'pms' => 'Piedmontese',
  'pnb' => 'Western Punjabi', 'ps' => 'Pashto', 'pt' => 'Portuguese', 'qu' => 'Quechua',
  'rm' => 'Romansh', 'ro' => 'Romanian', 'ru' => 'Russian', 'sa' => 'Sanskrit',
  'sah' => 'Yakut', 'sc' => 'Sardinian', 'scn' => 'Sicilian', 'sco' => 'Scots',
  'sd' => 'Sindhi', 'sh' => 'Serbo-Croatian', 'si' => 'Sinhalese', 'sk' => 'Slovak',
  'sl' => 'Slovenian', 'so' => 'Somali', 'sq' => 'Albanian', 'sr' => 'Serbian',
  'su' => 'Sundanese', 'sv' => 'Swedish', 'sw' => 'Swahili', 'ta' => 'Tamil',
  'te' => 'Telugu', 'tg' => 'Tajik', 'th' => 'Thai', 'tk' => 'Turkmen', 'tl' => 'Tagalog',
  'tr' => 'Turkish', 'tt' => 'Tatar', 'ug' => 'Uyghur', 'uk' => 'Ukrainian',
  'ur' => 'Urdu', 'uz' => 'Uzbek', 'vec' => 'Venetian', 'vi' => 'Vietnamese',
  'vls' => 'West Flemish', 'vo' => 'Volapük', 'wa' => 'Walloon', 'war' => 'Waray',
  'xmf' => 'Mingrelian', 'yi' => 'Yiddish', 'yo' => 'Yoruba', 'zea' => 'Zeelandic',
  'zh' => 'Chinese'
}.freeze

def get_sample_embedding(model_path, index)
  script = <<~PYTHON
    import sys
    import onnxruntime as ort
    import numpy as np

    try:
      sess = ort.InferenceSession('#{model_path}', providers=['CPUExecutionProvider'])
      input_name = sess.get_inputs()[0].name
      output_name = sess.get_outputs()[0].name

      embedding = sess.run([output_name], {input_name: np.array([#{index}], dtype=np.int64)})[0]

      stats = f"{np.mean(embedding):.6f},{np.std(embedding):.6f},{np.min(embedding):.6f},{np.max(embedding):.6f}"
      samples = ','.join([f'{v:.6f}' for v in embedding[0:10]])
      print(f"{stats}|{samples}")
      sys.exit(0)
    except Exception as e:
      print(f"ERROR:{e}", file=sys.stderr)
      sys.exit(1)
  PYTHON

  output, = Open3.capture3('python3', '-c', script)
  return output.strip
end

def generate_fixture_for_language(lang)
  model_path = "models/#{lang}/fasttext.#{lang}.onnx"
  metadata_path = "models/#{lang}/metadata.json"

  unless File.exist?(model_path)
    return { lang => { status: 'missing' } }
  end

  puts "  Generating #{lang.upcase}..."

  # Load metadata if exists
  vocab_size = 100_000
  if File.exist?(metadata_path)
    metadata = JSON.parse(File.read(metadata_path))
    vocab_size = metadata['vocab_size'] || 100_000
  end

  # Calculate test indices based on vocab size
  first_idx = 0
  mid_idx = vocab_size / 2
  last_idx = vocab_size - 1

  # Get sample embeddings
  samples = {}
  [first_idx, 1, [100, vocab_size - 1].max, [1000, vocab_size - 1].max, [10000, vocab_size - 1].max].each do |idx|
    result = get_sample_embedding(model_path, idx)
    if result.include?('ERROR')
      puts "    ⊗ Warning: Could not sample index #{idx}"
      next
    end

    parts = result.split('|')
    stats_parts = parts[0].split(',')
    sample_values = parts[1].split(',').map(&:to_f)

    samples[idx] = {
      mean: stats_parts[0].to_f,
      std: stats_parts[1].to_f,
      min: stats_parts[2].to_f,
      max: stats_parts[3].to_f,
      sample_values: sample_values.each_with_index.map { |i, v| { 'index' => i, 'value' => v } }
    }
  end

  # Create fixture
  fixture = {
    'metadata' => {
      'language_code' => lang,
      'language_name' => LANGUAGE_NAMES[lang] || lang.upcase,
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
      'min_mean' => -0.5,
      'max_mean' => 0.5,
      'max_std_dev' => 0.5,
      'check_finite' => true,
      'check_no_zeros' => false
    },
    'test_cases' => [
      {
        'name' => 'first_word',
        'input' => { 'word_index' => first_idx },
        'expected_output' => {
          'embedding_shape' => [300],
          'sample_values' => samples[first_idx][:sample_values][0..4].each_with_index.map { |i, v| { 'index' => i, 'value' => v, 'tolerance' => 0.01 } }
        }
      },
      {
        'name' => 'second_word',
        'input' => { 'word_index' => 1 },
        'expected_output' => {
          'embedding_shape' => [300],
          'sample_values' => samples[1][:sample_values][0..4].each_with_index.map { |i, v| { 'index' => i, 'value' => v, 'tolerance' => 0.01 } }
        }
      },
      {
        'name' => 'middle_word',
        'input' => { 'word_index' => mid_idx },
        'expected_output' => {
          'embedding_shape' => [300]
        }
      },
      {
        'name' => 'last_word',
        'input' => { 'word_index' => last_idx },
        'expected_output' => {
          'embedding_shape' => [300]
        }
      }
    ],
    'statistics' => {
      'vocab_sample' => {
        'indices' => samples.keys.sort,
        'expected_means' => samples.map do |idx, data|
          {
            'index' => idx,
            'mean' => data[:mean],
            'std' => data[:std],
            'tolerance' => 0.01
          }
        end
      }
    }
  }

  # Write fixture
  fixtures_dir = File.join(__dir__, '..', 'tests', 'fixtures')
  FileUtils.mkdir_p(fixtures_dir)

  File.write(File.join(fixtures_dir, "#{lang}.yaml"), fixture.to_yaml)

  { lang => { status: 'generated' } }
end

def main
  puts '=' * 80
  puts 'GENERATING TEST FIXTURES FOR ALL 157 LANGUAGES'
  puts '=' * 80
  puts

  results = {}
  generated = 0
  missing = 0

  ALL_LANGUAGES.each do |lang|
    result = generate_fixture_for_language(lang)
    results.merge!(result)

    if result[lang][:status] == 'generated'
      generated += 1
    else
      missing += 1
    end
  end

  puts
  puts '=' * 80
  puts "FIXTURE GENERATION COMPLETE"
  puts '=' * 80
  puts "Generated: #{generated}"
  puts "Missing: #{missing}"
  puts
  puts "Test fixtures created in: tests/fixtures/"
  puts
  puts "Usage:"
  puts "  python3 tests/test_onnx_model.py <lang>"
  puts "  Example: python3 tests/test_onnx_model.py af"
end

main if __FILE__ == $PROGRAM_NAME
