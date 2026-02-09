#!/bin/bash
# Test script for CE FastText ONNX model

echo "Testing CE..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ce
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ CE PASSED"
else
  echo "✗ CE FAILED"
fi

exit $exit_code
