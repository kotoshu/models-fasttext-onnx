#!/bin/bash
# Test script for HI FastText ONNX model

echo "Testing HI..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" hi
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ HI PASSED"
else
  echo "✗ HI FAILED"
fi

exit $exit_code
