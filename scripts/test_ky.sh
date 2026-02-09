#!/bin/bash
# Test script for KY FastText ONNX model

echo "Testing KY..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ky
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ KY PASSED"
else
  echo "✗ KY FAILED"
fi

exit $exit_code
