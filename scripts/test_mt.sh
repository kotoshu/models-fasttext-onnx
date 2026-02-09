#!/bin/bash
# Test script for MT FastText ONNX model

echo "Testing MT..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mt
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MT PASSED"
else
  echo "✗ MT FAILED"
fi

exit $exit_code
