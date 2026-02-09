#!/bin/bash
# Test script for SI FastText ONNX model

echo "Testing SI..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" si
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SI PASSED"
else
  echo "✗ SI FAILED"
fi

exit $exit_code
