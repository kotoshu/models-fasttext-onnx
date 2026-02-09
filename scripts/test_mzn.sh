#!/bin/bash
# Test script for MZN FastText ONNX model

echo "Testing MZN..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mzn
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MZN PASSED"
else
  echo "✗ MZN FAILED"
fi

exit $exit_code
