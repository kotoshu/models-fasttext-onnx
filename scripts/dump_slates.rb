$LOAD_PATH.unshift(File.expand_path("~/src/kotoshu/kotoshu/lib"))
require "kotoshu"
require "json"

lang = ARGV.fetch(0, "en")
infile = ARGV.fetch(1)
outfile = ARGV.fetch(2)

Kotoshu.setup(lang)
Kotoshu.instance_variable_set(:@spellcheckers, {})

File.open(outfile, "w") do |out|
  File.readlines(infile, chomp: true).each do |line|
    row = JSON.parse(line)
    typo = row["typo"]
    truth = row["correction"].downcase
    begin
      slate = Kotoshu.suggest(typo, language: lang, max_suggestions: 10).to_words.map(&:downcase)
    rescue StandardError
      slate = []
    end
    label = slate.index(truth) || -1
    out.puts(JSON.generate({ "typo" => typo, "slate" => slate, "label" => label,
                             "class" => row["class"] }))
  end
end
puts "dumped #{File.readlines(infile).size} composite slates"
