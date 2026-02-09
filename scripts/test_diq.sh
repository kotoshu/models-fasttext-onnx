#!/bin/bash
# Test script for DIQ FastText ONNX model

echo "Testing DIQ..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" diq
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ DIQ PASSED"
else
  echo "✗ DIQ FAILED"
fi

exit $exit_code
