#!/bin/bash
# Test script for UG FastText ONNX model

echo "Testing UG..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ug
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ UG PASSED"
else
  echo "✗ UG FAILED"
fi

exit $exit_code
