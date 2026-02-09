#!/bin/bash
# Test script for LI FastText ONNX model

echo "Testing LI..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" li
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ LI PASSED"
else
  echo "✗ LI FAILED"
fi

exit $exit_code
