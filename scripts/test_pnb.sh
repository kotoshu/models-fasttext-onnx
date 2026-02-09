#!/bin/bash
# Test script for PNB FastText ONNX model

echo "Testing PNB..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" pnb
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ PNB PASSED"
else
  echo "✗ PNB FAILED"
fi

exit $exit_code
