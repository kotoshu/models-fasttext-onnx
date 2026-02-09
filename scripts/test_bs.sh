#!/bin/bash
# Test script for BS FastText ONNX model

echo "Testing BS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" bs
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BS PASSED"
else
  echo "✗ BS FAILED"
fi

exit $exit_code
