#!/bin/bash
# Test script for ALS FastText ONNX model

echo "Testing ALS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" als
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ ALS PASSED"
else
  echo "✗ ALS FAILED"
fi

exit $exit_code
