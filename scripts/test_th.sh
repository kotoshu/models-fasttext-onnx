#!/bin/bash
# Test script for TH FastText ONNX model

echo "Testing TH..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" th
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ TH PASSED"
else
  echo "✗ TH FAILED"
fi

exit $exit_code
