#!/bin/bash
# Test script for BE FastText ONNX model

echo "Testing BE..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" be
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BE PASSED"
else
  echo "✗ BE FAILED"
fi

exit $exit_code
