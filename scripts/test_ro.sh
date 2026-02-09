#!/bin/bash
# Test script for RO FastText ONNX model

echo "Testing RO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ro
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ RO PASSED"
else
  echo "✗ RO FAILED"
fi

exit $exit_code
