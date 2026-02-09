#!/bin/bash
# Test script for LT FastText ONNX model

echo "Testing LT..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" lt
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ LT PASSED"
else
  echo "✗ LT FAILED"
fi

exit $exit_code
