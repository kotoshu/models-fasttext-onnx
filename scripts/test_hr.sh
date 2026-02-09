#!/bin/bash
# Test script for HR FastText ONNX model

echo "Testing HR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" hr
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ HR PASSED"
else
  echo "✗ HR FAILED"
fi

exit $exit_code
