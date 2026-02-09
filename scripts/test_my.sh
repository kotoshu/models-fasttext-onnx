#!/bin/bash
# Test script for MY FastText ONNX model

echo "Testing MY..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" my
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MY PASSED"
else
  echo "✗ MY FAILED"
fi

exit $exit_code
