#!/bin/bash
# Test script for NAH FastText ONNX model

echo "Testing NAH..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" nah
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ NAH PASSED"
else
  echo "✗ NAH FAILED"
fi

exit $exit_code
