#!/bin/bash
# Test script for BN FastText ONNX model

echo "Testing BN..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" bn
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BN PASSED"
else
  echo "✗ BN FAILED"
fi

exit $exit_code
