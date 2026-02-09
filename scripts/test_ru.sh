#!/bin/bash
# Test script for RU FastText ONNX model

echo "Testing RU..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ru
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ RU PASSED"
else
  echo "✗ RU FAILED"
fi

exit $exit_code
