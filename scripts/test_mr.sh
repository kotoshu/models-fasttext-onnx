#!/bin/bash
# Test script for MR FastText ONNX model

echo "Testing MR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mr
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MR PASSED"
else
  echo "✗ MR FAILED"
fi

exit $exit_code
