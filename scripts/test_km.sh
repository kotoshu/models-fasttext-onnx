#!/bin/bash
# Test script for KM FastText ONNX model

echo "Testing KM..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" km
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ KM PASSED"
else
  echo "✗ KM FAILED"
fi

exit $exit_code
