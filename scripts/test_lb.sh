#!/bin/bash
# Test script for LB FastText ONNX model

echo "Testing LB..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" lb
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ LB PASSED"
else
  echo "✗ LB FAILED"
fi

exit $exit_code
