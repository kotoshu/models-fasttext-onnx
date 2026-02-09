#!/bin/bash
# Test script for VI FastText ONNX model

echo "Testing VI..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" vi
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ VI PASSED"
else
  echo "✗ VI FAILED"
fi

exit $exit_code
