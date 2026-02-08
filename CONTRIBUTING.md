# Contributing to FastText ONNX Models

Thank you for your interest in contributing to this repository!

## Overview

This repository contains FastText word embedding models converted to ONNX format for use with Kotoshu spell checker. Contributions are welcome in the following areas:

- Adding new language models
- Improving conversion scripts
- Bug fixes and optimizations
- Documentation improvements

## Setting Up Your Development Environment

### Prerequisites

```bash
# Python 3.8+ required
python3 --version

# Install dependencies
pip install onnx onnxruntime numpy
```

### Cloning the Repository

```bash
# Clone with Git LFS (required for large model files)
git clone https://github.com/kotoshu/models-fasttext-onnx.git
cd models-fasttext-onnx
git lfs install
git lfs pull
```

## Adding a New Language Model

### 1. Download FastText Vectors

Download the FastText vectors for your language from https://fasttext.cc/docs/en/crawl-vectors.html:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz
gunzip cc.{lang}.300.vec.gz
```

### 2. Convert to ONNX

Use the provided conversion script:

```bash
python3 scripts/fasttext_to_onnx.py \
    cc.{lang}.300.vec \
    models/{lang}/fasttext.{lang}.onnx \
    --vocab-size 100000
```

### 3. Verify the Model

```bash
python3 scripts/test_onnx.py {lang}
```

### 4. Create Metadata

Create `models/{lang}/metadata.json`:

```json
{
  "version": "YYYY-MM-DDTHH:MM:SSZ",
  "language": "{lang}",
  "type": "onnx",
  "file": "fasttext.{lang}.onnx",
  "checksum": "sha256_hash_here",
  "source_model": "cc.{lang}.300.vec",
  "conversion_method": "fasttext_to_onnx.py",
  "opset_version": 11,
  "vocab_size": 100000,
  "embedding_dim": 300
}
```

### 5. Test with Kotoshu

```bash
# Add to Kotoshu's ModelCache
kotoshu cache test models/{lang}/fasttext.{lang}.onnx
```

## Conversion Script Guidelines

When modifying `scripts/fasttext_to_onnx.py`:

- **Maintain ONNX compatibility**: Use opset version 11 and IR version 11
- **Preserve semantics**: Ensure embeddings match the original FastText vectors
- **Add tests**: Verify conversion accuracy with numerical comparison tests
- **Document changes**: Update this README with any breaking changes

### ONNX Model Requirements

All ONNX models must:

1. **Use IR version 11** and **opset version 11** for onnxruntime 1.23.2+ compatibility
2. **Have correct input/output shapes**: `word_index` (int64[1]) → `embedding` (float32[300])
3. **Include metadata**: Add language, version, and conversion info to model metadata_props
4. **Pass functional tests**: Load with onnxruntime and return valid 300D embeddings

## Testing

### Running Tests

```bash
# Test all models
ruby scripts/verify_all_models.rb

# Test individual model
python3 scripts/test_onnx.py en

# Test conversion accuracy
python3 scripts/test_conversion_accuracy.py en
```

### Accuracy Verification

When converting a new model, verify embeddings match the original:

```python
import numpy as np
from gensim.models import FastText  # Original FastText

# Load original FastText model
original_model = FastText.load_fasttext_format('cc.en.300.bin')

# Load ONNX model
import onnxruntime as ort
sess = ort.InferenceSession('models/en/fasttext.en.onnx')

# Compare embeddings
word = "hello"
original_emb = original_model.wv[word]
onnx_emb = sess.run(None, {sess.get_inputs()[0].name: [word_index]})[0]

# Should be nearly identical (allowing for floating point precision)
assert np.allclose(original_emb, onnx_emb, rtol=1e-5)
```

## Pull Request Process

1. **Fork the repository** and create a new branch
2. **Make your changes** following the guidelines above
3. **Test thoroughly** using the provided test scripts
4. **Submit a pull request** with:
   - Clear description of changes
   - Test results showing all tests pass
   - Screenshot/example of the model working
   - Updated documentation if needed

## Code Style

### Python

- Follow PEP 8 guidelines
- Use 4 spaces for indentation
- Add docstrings to all functions
- Type hints where appropriate

### Ruby

- Follow RuboCop style guidelines
- Use 2 spaces for indentation
- Frozen string literals: `# frozen_string_literal: true`

## Documentation

- Update README.md when adding new languages
- Add inline comments for complex logic
- Keep CHANGELOG.md updated with version changes
- Document any breaking changes prominently

## Issue Reporting

When reporting issues, please include:

1. **Environment details**: Python version, onnxruntime version
2. **Language code**: Which language model has the issue
3. **Error message**: Full error traceback
4. **Steps to reproduce**: Minimal reproduction case
5. **Expected vs actual**: What you expected vs what happened

## License

By contributing, you agree that your contributions will be licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License.

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others when you can
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## Questions?

- Open an issue on GitHub
- Email: support@kotoshu.io
- Discussion: [GitHub Discussions](../../discussions)

Thank you for contributing! 🎉
