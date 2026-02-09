#!/bin/bash
# Test script for SU FastText ONNX model

echo "Testing SU..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" su
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SU PASSED"
else
  echo "✗ SU FAILED"
fi

exit $exit_code
