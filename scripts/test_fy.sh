#!/bin/bash
# Test script for FY FastText ONNX model

echo "Testing FY..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" fy
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ FY PASSED"
else
  echo "✗ FY FAILED"
fi

exit $exit_code
