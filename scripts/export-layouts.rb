$LOAD_PATH.unshift(File.expand_path("lib", Dir.pwd))
require "kotoshu"
require "json"

layouts = Kotoshu::Keyboard::Registry.available_layouts.map do |l|
  {
    "name" => l.name,
    "language_codes" => l.language_codes,
    "key_positions" => l.key_positions
  }
end
File.write("/tmp/layout-grids.json", JSON.pretty_generate({ "layouts" => layouts }))
puts "exported #{layouts.size} layouts"
