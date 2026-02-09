#!/bin/bash
# Test script for EL FastText ONNX model

echo "Testing EL..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" el
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ EL PASSED"
else
  echo "✗ EL FAILED"
fi

exit $exit_code
