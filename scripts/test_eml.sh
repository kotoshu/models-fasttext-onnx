#!/bin/bash
# Test script for EML FastText ONNX model

echo "Testing EML..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" eml
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ EML PASSED"
else
  echo "✗ EML FAILED"
fi

exit $exit_code
