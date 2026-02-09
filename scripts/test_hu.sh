#!/bin/bash
# Test script for HU FastText ONNX model

echo "Testing HU..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" hu
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ HU PASSED"
else
  echo "✗ HU FAILED"
fi

exit $exit_code
