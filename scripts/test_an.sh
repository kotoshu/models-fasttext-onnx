#!/bin/bash
# Test script for AN FastText ONNX model

echo "Testing AN..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" an
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ AN PASSED"
else
  echo "✗ AN FAILED"
fi

exit $exit_code
