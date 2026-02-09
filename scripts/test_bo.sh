#!/bin/bash
# Test script for BO FastText ONNX model

echo "Testing BO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" bo
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BO PASSED"
else
  echo "✗ BO FAILED"
fi

exit $exit_code
