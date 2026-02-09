#!/bin/bash
# Test script for TL FastText ONNX model

echo "Testing TL..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" tl
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ TL PASSED"
else
  echo "✗ TL FAILED"
fi

exit $exit_code
