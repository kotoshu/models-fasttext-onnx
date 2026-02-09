#!/bin/bash
# Test script for SK FastText ONNX model

echo "Testing SK..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sk
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SK PASSED"
else
  echo "✗ SK FAILED"
fi

exit $exit_code
