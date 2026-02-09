#!/bin/bash
# Test script for MYV FastText ONNX model

echo "Testing MYV..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" myv
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MYV PASSED"
else
  echo "✗ MYV FAILED"
fi

exit $exit_code
