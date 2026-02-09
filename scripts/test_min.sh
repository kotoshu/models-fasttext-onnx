#!/bin/bash
# Test script for MIN FastText ONNX model

echo "Testing MIN..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" min
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MIN PASSED"
else
  echo "✗ MIN FAILED"
fi

exit $exit_code
