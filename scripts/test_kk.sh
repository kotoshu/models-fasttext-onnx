#!/bin/bash
# Test script for KK FastText ONNX model

echo "Testing KK..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" kk
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ KK PASSED"
else
  echo "✗ KK FAILED"
fi

exit $exit_code
