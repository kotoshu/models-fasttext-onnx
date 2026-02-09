#!/bin/bash
# Test script for MN FastText ONNX model

echo "Testing MN..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mn
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MN PASSED"
else
  echo "✗ MN FAILED"
fi

exit $exit_code
