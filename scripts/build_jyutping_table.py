#!/usr/bin/env python3
"""S8-E4: jyutping homophone table for zh-Hant-HK (TODO.sota-impl/11).

Source: Unicode Unihan_Readings.txt (kCantonese), the same
char -> syllable -> homophone-char shape as the S4 pinyin tables
(build_confusion_tables.py). Tones are kept — the pinyin table keeps
tones for syllables 1-4 and only strips neutral 5, so tone-preserving
groups are the house convention. Output feeds the generator's
ime-confusion class via /tmp/cjk/jyutping.txt (char<TAB>syl lines) and
is committed to eval/confusion/ as the frozen derivation.
"""
import json
import re
from collections import defaultdict
from pathlib import Path
import urllib.request

UCD = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
TMP = Path("/tmp/cjk")
OUT_EVIDENCE = Path("eval/confusion/jyutping.json")

line_re = re.compile(r"^U\+([0-9A-Fa-f]{4,6})\tkCantonese\t(.+)$")

groups = defaultdict(set)
if not (TMP / "Unihan_Readings.txt").exists():
    import zipfile
    import io
    req = urllib.request.Request(UCD, headers={"User-Agent": "kotoshu-s8e4/1.0"})
    blob = urllib.request.urlopen(req).read()
    (TMP / "Unihan.zip").write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        (TMP / "Unihan_Readings.txt").write_bytes(zf.read("Unihan_Readings.txt"))

for line in (TMP / "Unihan_Readings.txt").read_text(encoding="utf-8").splitlines():
    m = line_re.match(line)
    if not m:
        continue
    ch = chr(int(m.group(1), 16))
    for syl in m.group(2).split():
        syl = syl.strip()
        if syl:
            groups[ch].add(syl)

by_syl = defaultdict(list)
for ch, syls in sorted(groups.items()):
    for s in sorted(syls):
        by_syl[s].append(ch)

with open(TMP / "jyutping.txt", "w", encoding="utf-8") as fh:
    for ch, syls in sorted(groups.items()):
        for s in sorted(syls):
            fh.write(f"{ch}\t{s}\n")

covered = {c for e in json.loads(
    Path.home().joinpath(".cache/kotoshu/frequency-lists/zh-Hant-HK/frequency.json").read_text()
)["full_list"][:10_000] for c in e["word"] if ord(c) >= 0x3400}
hit = sum(1 for c in covered if c in groups)
doc = {
    "spec": "kotoshu.confusion-table/v1",
    "language": "zh-Hant-HK",
    "source": "Unicode Unihan kCantonese (UCD; Unicode License)",
    "source_url": UCD,
    "chars_with_readings": len(groups),
    "syllables": len(by_syl),
    "top10k_wordlist_char_coverage": {
        "covered": hit, "total": len(covered),
        "ratio": round(hit / max(1, len(covered)), 4),
    },
    "homophones": {s: chars for s, chars in sorted(by_syl.items())
                   if len(chars) > 1},
}
OUT_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
OUT_EVIDENCE.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
print(f"chars={len(groups)} syllables={len(by_syl)} "
      f"top10k coverage={hit}/{len(covered)} "
      f"({doc['top10k_wordlist_char_coverage']['ratio']:.1%})")
