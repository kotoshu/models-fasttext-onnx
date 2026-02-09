#!/bin/bash
# Test script for SH FastText ONNX model

echo "Testing SH..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sh
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SH PASSED"
else
  echo "✗ SH FAILED"
fi

exit $exit_code
