#!/bin/bash
# Test script for AM FastText ONNX model

echo "Testing AM..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" am
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ AM PASSED"
else
  echo "✗ AM FAILED"
fi

exit $exit_code
