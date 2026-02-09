#!/bin/bash
# Test script for DV FastText ONNX model

echo "Testing DV..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" dv
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ DV PASSED"
else
  echo "✗ DV FAILED"
fi

exit $exit_code
