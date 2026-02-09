#!/bin/bash
# Test script for OR FastText ONNX model

echo "Testing OR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" or
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ OR PASSED"
else
  echo "✗ OR FAILED"
fi

exit $exit_code
