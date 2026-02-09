#!/bin/bash
# Test script for AR FastText ONNX model

echo "Testing AR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ar
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ AR PASSED"
else
  echo "✗ AR FAILED"
fi

exit $exit_code
