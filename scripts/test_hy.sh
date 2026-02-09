#!/bin/bash
# Test script for HY FastText ONNX model

echo "Testing HY..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" hy
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ HY PASSED"
else
  echo "✗ HY FAILED"
fi

exit $exit_code
