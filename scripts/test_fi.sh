#!/bin/bash
# Test script for FI FastText ONNX model

echo "Testing FI..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" fi
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ FI PASSED"
else
  echo "✗ FI FAILED"
fi

exit $exit_code
