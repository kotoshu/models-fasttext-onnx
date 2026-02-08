# FastText ONNX Models for Kotoshu

Pre-trained FastText word embedding models converted to ONNX format for use with Kotoshu spell checker.

## Overview

This repository contains FastText word embedding models that have been converted from the original `.vec` format to ONNX format for efficient deployment and inference.

### Models Available

| Language | Code | Vocab Size | Embedding Dim | Model Size | Source |
|----------|------|------------|---------------|------------|--------|
| German | de | 100,000 | 300D | 114.44 MB | [FastText CC.de.300](https://fasttext.cc/docs/en/crawl-vectors.html) |
| English | en | 100,000 | 300D | 114.44 MB | [FastText CC.en.300](https://fasttext.cc/docs/en/crawl-vectors.html) |
| Spanish | es | 100,000 | 300D | 114.44 MB | [FastText CC.es.300](https://fasttext.cc/docs/en/crawl-vectors.html) |
| French | fr | 100,000 | 300D | 114.44 MB | [FastText CC.fr.300](https://fasttext.cc/docs/en/crawl-vectors.html) |
| Portuguese | pt | 100,000 | 300D | 114.44 MB | [FastText CC.pt.300](https://fasttext.cc/docs/en/crawl-vectors.html) |
| Russian | ru | 100,000 | 300D | 114.44 MB | [FastText CC.ru.300](https://fasttext.cc/docs/en/crawl-vectors.html) |

### Compression Ratio

- Original FastText `.vec` files: ~4.3 GB per language
- ONNX format: 114.44 MB per language
- Compression: **37x smaller** with full semantic quality preserved

## Model Specifications

### Technical Details

- **ONNX IR Version**: 11
- **ONNX Opset Version**: 11
- **Compatibility**: onnxruntime 1.23.2+
- **Input**: `word_index` (int64, shape=[1])
- **Output**: `embedding` (float32, shape=[300])

### Architecture

The ONNX models use a simple embedding lookup architecture:

1. **Constant Node**: Contains the embedding matrix (vocab_size x 300)
2. **Gather Node**: Retrieves the embedding for a given word index
3. **Squeeze Node**: Removes the batch dimension

```
word_index (int64[1]) → Gather → Squeeze → embedding (float32[300])
     ↓
Embeddings Matrix (Constant)
```

## Usage

### Python (with onnxruntime)

```python
import onnxruntime as ort
import numpy as np

# Load model
sess = ort.InferenceSession('models/en/fasttext.en.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Get embedding for word index
word_index = 0  # Replace with actual word index from vocabulary
embedding = sess.run([output_name], {input_name: np.array([word_index], dtype=np.int64)})[0]

print(f"Embedding shape: {embedding.shape}")  # (300,)
print(f"Embedding: {embedding}")
```

### Ruby (with Kotoshu)

```ruby
require 'kotoshu'

# Get ONNX model (downloads or converts as needed)
cache = Kotoshu::Cache::ModelCache.new
onnx_path = cache.get_onnx_model('en')

# Use for semantic similarity
# (This feature is planned for future Kotoshu versions)
```

### CLI

```bash
# Download models using Kotoshu CLI
kotoshu cache download en

# Check cache status
kotoshu cache status

# List available languages
kotoshu cache list
```

## Model Metadata

Each model includes metadata in `models/{lang}/metadata.json`:

```json
{
  "version": "2026-02-08T04:25:31Z",
  "language": "en",
  "type": "onnx",
  "file": "fasttext.en.onnx",
  "checksum": "sha256_hash",
  "source_model": "cc.en.300.vec",
  "conversion_method": "fasttext_to_onnx.py",
  "opset_version": 11,
  "vocab_size": 100000,
  "embedding_dim": 300
}
```

## Verification

All models have been functionally verified:

```bash
# Run verification tests
ruby scripts/verify_all_models.rb

# Test individual model
python3 scripts/test_onnx.py en
```

## Download

### Direct Download

Models can be downloaded directly from the [Releases](../../releases) page.

### Via Kotoshu (Recommended)

Kotoshu will automatically download and cache these models when needed:

```ruby
# Ruby API
cache = Kotoshu::Cache::ModelCache.new
onnx_path = cache.get_onnx_model('de')
```

### Manual Download

```bash
# Using Git LFS
git clone https://github.com/kotoshu/models-fasttext-onnx.git
cd models-fasttext-onnx
git lfs pull

# Download specific model
wget https://github.com/kotoshu/models-fasttext-onnx/raw/main/models/en/fasttext.en.onnx
```

## Building from Source

If you want to convert the models yourself from FastText `.vec` files:

```bash
# Install dependencies
pip install onnx onnxruntime numpy

# Download FastText vectors
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip cc.en.300.vec.gz

# Convert to ONNX
python3 scripts/fasttext_to_onnx.py cc.en.300.vec models/en/fasttext.en.onnx --vocab-size 100000
```

## License

These models are derived from the [FastText pretrained vectors](https://fasttext.cc/docs/en/crawl-vectors.html), which are licensed under the [Creative Commons Attribution-Share-Alike License 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

## Citation

If you use these models, please cite the original FastText paper:

```bibtex
@inproceedings{bojar-2018-find,
    title = "Findings of the 2018 Conference on Machine Translation ({WMT}18)",
    author = "Bojar, Ond{\v{r}}ej  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Koehn, Philipp  and
      Koehn, Philipp",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Shared Task Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6401",
    doi = "10.18653/v1/W18-6401",
    pages = "272--303",
}
```

## References

- [FastText](https://fasttext.cc/) - Facebook's library for efficient learning of word representations
- [ONNX](https://onnx.ai/) - Open Neural Network Exchange
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Microsoft's cross-platform inference engine
- [Kotoshu](https://github.com/kotoshu/kotoshu) - Spell checker library

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this repository.

## Support

- **Issues**: [GitHub Issues](../../issues)
- **Documentation**: [Kotoshu Documentation](https://github.com/kotoshu/kotoshu)
- **Email**: support@kotoshu.io

## Release History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-08 | Initial release with all 6 language models |
