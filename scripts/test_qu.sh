#!/bin/bash
# Test script for QU FastText ONNX model

echo "Testing QU..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" qu
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ QU PASSED"
else
  echo "✗ QU FAILED"
fi

exit $exit_code
