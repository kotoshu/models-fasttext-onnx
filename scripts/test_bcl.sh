#!/bin/bash
# Test script for BCL FastText ONNX model

echo "Testing BCL..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" bcl
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BCL PASSED"
else
  echo "✗ BCL FAILED"
fi

exit $exit_code
