#!/bin/bash
# Test script for NEW FastText ONNX model

echo "Testing NEW..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" new
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ NEW PASSED"
else
  echo "✗ NEW FAILED"
fi

exit $exit_code
