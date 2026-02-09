#!/bin/bash
# Test script for OS FastText ONNX model

echo "Testing OS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" os
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ OS PASSED"
else
  echo "✗ OS FAILED"
fi

exit $exit_code
