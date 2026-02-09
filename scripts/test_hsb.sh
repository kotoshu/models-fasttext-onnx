#!/bin/bash
# Test script for HSB FastText ONNX model

echo "Testing HSB..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" hsb
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ HSB PASSED"
else
  echo "✗ HSB FAILED"
fi

exit $exit_code
