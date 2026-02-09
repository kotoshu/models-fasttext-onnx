#!/bin/bash
# Test script for PL FastText ONNX model

echo "Testing PL..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" pl
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ PL PASSED"
else
  echo "✗ PL FAILED"
fi

exit $exit_code
