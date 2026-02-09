#!/bin/bash
# Test script for GOM FastText ONNX model

echo "Testing GOM..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" gom
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ GOM PASSED"
else
  echo "✗ GOM FAILED"
fi

exit $exit_code
