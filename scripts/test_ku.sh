#!/bin/bash
# Test script for KU FastText ONNX model

echo "Testing KU..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ku
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ KU PASSED"
else
  echo "✗ KU FAILED"
fi

exit $exit_code
