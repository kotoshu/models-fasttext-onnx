#!/bin/bash
# Test script for GU FastText ONNX model

echo "Testing GU..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" gu
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ GU PASSED"
else
  echo "✗ GU FAILED"
fi

exit $exit_code
