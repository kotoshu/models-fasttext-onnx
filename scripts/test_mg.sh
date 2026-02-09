#!/bin/bash
# Test script for MG FastText ONNX model

echo "Testing MG..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mg
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MG PASSED"
else
  echo "✗ MG FAILED"
fi

exit $exit_code
