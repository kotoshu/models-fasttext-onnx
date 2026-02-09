#!/bin/bash
# Test script for VLS FastText ONNX model

echo "Testing VLS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" vls
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ VLS PASSED"
else
  echo "✗ VLS FAILED"
fi

exit $exit_code
