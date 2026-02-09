#!/bin/bash
# Test script for KN FastText ONNX model

echo "Testing KN..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" kn
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ KN PASSED"
else
  echo "✗ KN FAILED"
fi

exit $exit_code
