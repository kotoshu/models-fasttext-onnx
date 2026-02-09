#!/bin/bash
# Test script for BH FastText ONNX model

echo "Testing BH..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" bh
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BH PASSED"
else
  echo "✗ BH FAILED"
fi

exit $exit_code
