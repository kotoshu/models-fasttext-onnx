#!/bin/bash
# Test script for SW FastText ONNX model

echo "Testing SW..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sw
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SW PASSED"
else
  echo "✗ SW FAILED"
fi

exit $exit_code
