#!/bin/bash
# Test script for GD FastText ONNX model

echo "Testing GD..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" gd
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ GD PASSED"
else
  echo "✗ GD FAILED"
fi

exit $exit_code
