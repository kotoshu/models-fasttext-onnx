#!/bin/bash
# Test script for YI FastText ONNX model

echo "Testing YI..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" yi
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ YI PASSED"
else
  echo "✗ YI FAILED"
fi

exit $exit_code
