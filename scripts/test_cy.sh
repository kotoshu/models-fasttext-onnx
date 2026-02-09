#!/bin/bash
# Test script for CY FastText ONNX model

echo "Testing CY..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" cy
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ CY PASSED"
else
  echo "✗ CY FAILED"
fi

exit $exit_code
