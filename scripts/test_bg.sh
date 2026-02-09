#!/bin/bash
# Test script for BG FastText ONNX model

echo "Testing BG..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" bg
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BG PASSED"
else
  echo "✗ BG FAILED"
fi

exit $exit_code
