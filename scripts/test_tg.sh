#!/bin/bash
# Test script for TG FastText ONNX model

echo "Testing TG..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" tg
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ TG PASSED"
else
  echo "✗ TG FAILED"
fi

exit $exit_code
