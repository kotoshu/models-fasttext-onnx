# ALL 157 FASTTEXT LANGUAGES - Conversion Plan

## Overview

FastText provides pre-trained word vectors for **157 languages**. This document outlines the plan to convert ALL of them to ONNX format for the Kotoshu spell checker.

## Current Status

### Completed (6 languages)
- German (de)
- English (en)
- Spanish (es)
- French (fr)
- Portuguese (pt)
- Russian (ru)

### Remaining (151 languages)
All other languages from the FastText crawl vectors dataset.

## Complete Language List

| # | Code | Language | # | Code | Language | # | Code | Language |
|---|------|----------|---|------|----------|---|------|----------|
| 1 | af | Afrikaans | 54 | he | Hebrew | 107 | sa | Sanskrit |
| 2 | als | Alemannic | 55 | hi | Hindi | 108 | sah | Sakha |
| 3 | am | Amharic | 56 | hif | Fiji Hindi | 109 | sc | Sardinian |
| 4 | an | Aragonese | 57 | hr | Croatian | 110 | scn | Sicilian |
| 5 | ar | Arabic | 58 | hsb | Upper Sorbian | 111 | sco | Scots |
| 6 | arz | Egyptian Arabic | 59 | ht | Haitian | 112 | sd | Sindhi |
| 7 | as | Assamese | 60 | hu | Hungarian | 113 | sh | Serbo-Croatian |
| 8 | ast | Asturian | 61 | hy | Armenian | 114 | si | Sinhalese |
| 9 | az | Azerbaijani | 62 | ia | Interlingua | 115 | sk | Slovak |
| 10 | azb | Southern Azerbaijani | 63 | id | Indonesian | 116 | sl | Slovenian |
| 11 | ba | Bashkir | 64 | ilo | Ilokano | 117 | so | Somali |
| 12 | bar | Bavarian | 65 | io | Ido | 118 | sq | Albanian |
| 13 | bcl | Central Bicolano | 66 | is | Icelandic | 119 | sr | Serbian |
| 14 | be | Belarusian | 67 | it | Italian | 120 | su | Sundanese |
| 15 | bg | Bulgarian | 68 | ja | Japanese | 121 | sv | Swedish |
| 16 | bh | Bihari | 69 | jv | Javanese | 122 | sw | Swahili |
| 17 | bn | Bengali | 70 | ka | Georgian | 123 | ta | Tamil |
| 18 | bo | Tibetan | 71 | kk | Kazakh | 124 | te | Telugu |
| 19 | bpy | Bishnupriya Manipuri | 72 | km | Khmer | 125 | tg | Tajik |
| 20 | br | Breton | 73 | kn | Kannada | 126 | th | Thai |
| 21 | bs | Bosnian | 74 | ko | Korean | 127 | tk | Turkmen |
| 22 | ca | Catalan | 75 | ku | Kurdish (Kurmanji) | 128 | tl | Tagalog |
| 23 | ce | Chechen | 76 | ky | Kirghiz | 129 | tr | Turkish |
| 24 | ceb | Cebuano | 77 | la | Latin | 130 | tt | Tatar |
| 25 | ckb | Kurdish (Sorani) | 78 | lb | Luxembourgish | 131 | ug | Uyghur |
| 26 | co | Corsican | 79 | li | Limburgish | 132 | uk | Ukrainian |
| 27 | cs | Czech | 80 | lmo | Lombard | 133 | ur | Urdu |
| 28 | cv | Chuvash | 81 | lt | Lithuanian | 134 | uz | Uzbek |
| 29 | cy | Welsh | 82 | lv | Latvian | 135 | vec | Venetian |
| 30 | da | Danish | 83 | mai | Maithili | 136 | vi | Vietnamese |
| 31 | **de** | **German ✓** | 84 | mg | Malagasy | 137 | vls | West Flemish |
| 32 | diq | Zazaki | 85 | mhr | Meadow Mari | 138 | vo | Volapük |
| 33 | dv | Divehi | 86 | min | Minangkabau | 139 | wa | Walloon |
| 34 | el | Greek | 87 | mk | Macedonian | 140 | war | Waray |
| 35 | eml | Emilian-Romagnol | 88 | ml | Malayalam | 141 | xmf | Mingrelian |
| 36 | **en** | **English ✓** | 89 | mn | Mongolian | 142 | yi | Yiddish |
| 37 | eo | Esperanto | 90 | mr | Marathi | 143 | yo | Yoruba |
| 38 | **es** | **Spanish ✓** | 91 | mrj | Hill Mari | 144 | zea | Zeelandic |
| 39 | et | Estonian | 92 | ms | Malay | 145 | zh | Chinese |
| 40 | eu | Basque | 93 | mt | Maltese | | | |
| 41 | fa | Persian | 94 | mwl | Mirandese | | | |
| 42 | fi | Finnish | 95 | my | Burmese | | | |
| 43 | **fr** | **French ✓** | 96 | myv | Erzya | | | |
| 44 | frr | North Frisian | 97 | mzn | Mazandarani | | | |
| 45 | fy | West Frisian | 98 | nah | Nahuatl | | | |
| 46 | ga | Irish | 99 | nap | Neapolitan | | | |
| 47 | gd | Scottish Gaelic | 100 | nds | Low Saxon | | | |
| 48 | gl | Galician | 101 | ne | Nepali | | | |
| 49 | gom | Goan Konkani | 102 | new | Newar | | | |
| 50 | gu | Gujarati | 103 | nl | Dutch | | | |
| 51 | gv | Manx | 104 | nn | Norwegian (Nynorsk) | | | |
| 52 | ha | (not in crawl) | 105 | no | Norwegian (Bokmål) | | | |
| 53 | **hr** | **Croatian** | 106 | nso | Northern Sotho | | | |

## Resource Requirements

### Storage
- **Temporary**: ~700 GB (downloads + processing)
- **Final**: ~18 GB (ONNX models, 114 MB × 157 languages)
- **Recommended**: 1 TB free disk space

### Bandwidth
- **Total download**: ~700 GB of .vec.gz files
- **At 100 Mbps**: ~20 hours
- **At 1 Gbps**: ~2 hours

### Processing Time
- **Download**: 2-20 hours (depending on bandwidth)
- **Conversion**: ~5-10 minutes per language
- **Total**: 20-40 hours

### Memory
- **Minimum**: 8 GB RAM
- **Recommended**: 16 GB RAM

## Conversion Script

A fully automated script is provided:

```bash
cd /Users/mulgogi/src/kotoshu/models-fasttext-onnx
ruby scripts/deploy_all_157.rb
```

This script will:
1. Download all 157 FastText .vec.gz files
2. Convert each to ONNX format (100K vocabulary)
3. Verify each model works with onnxruntime
4. Generate metadata files
5. Clean up temporary files

## Model Specifications (Per Language)

- **Source**: FastText Common Crawl vectors (cc.{lang}.300.vec.gz)
- **Vocabulary**: Top 100,000 words (from 2M total)
- **Embeddings**: 300-dimensional vectors
- **Format**: ONNX IR v11, opset v11
- **Size**: ~114 MB per model (37x compression)
- **Compatibility**: onnxruntime 1.23.2+

## Deployment Strategy

### Phase 1: Core Languages (DONE)
High-priority languages with widespread use:
- de, en, es, fr, pt, ru (6 languages, 686 MB)

### Phase 2: Major Languages (NEXT)
Top 20 languages by speaker count:
- zh, hi, es, fr, ar, ru, pt, id, ja, de, jv, ko, fr, tr, vi, it, fa, pl, uk, nl

### Phase 3: Regional Languages
European and regional languages:
- it, nl, pl, cs, sv, da, fi, no, hu, ro, el, cs, etc.

### Phase 4: All Remaining Languages
All 157 languages from FastText.

## Progress Tracking

Current progress can be tracked with:

```bash
# Count completed models
ls models/*/*.onnx | wc -l

# Check status
ruby scripts/verify_all_models.rb
```

## GitHub Storage

With all 157 languages:
- **Total size**: ~18 GB
- **File count**: 157 .onnx files + 157 metadata.json files
- **Git LFS**: Required for large files

**Note**: GitHub has a 100 GB soft limit for Git LFS. We may need to:
1. Use GitHub Packages for releases
2. Create multiple repositories by region
3. Use alternative hosting (S3, Azure Blob, etc.)

## Next Steps

1. **Run the conversion script**:
   ```bash
   ruby scripts/deploy_all_157.rb
   ```

2. **Monitor progress**: Check terminal output for status

3. **Verify models**:
   ```bash
   ruby scripts/verify_all_models.rb
   ```

4. **Update README**: Add language list and statistics

5. **Push to GitHub**: May need to split into multiple pushes

## References

- FastText crawl vectors: https://fasttext.cc/docs/en/crawl-vectors.html
- Source paper: "Learning Word Vectors for 157 Languages" (Grave et al., 2018)
- ONNX format: https://onnx.ai/
- onnxruntime: https://github.com/microsoft/onnxruntime
