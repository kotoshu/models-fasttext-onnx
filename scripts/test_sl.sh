#!/bin/bash
# Test script for SL FastText ONNX model

echo "Testing SL..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sl
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SL PASSED"
else
  echo "✗ SL FAILED"
fi

exit $exit_code
