#!/bin/bash
# Test script for AF FastText ONNX model

echo "Testing AF..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" af
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ AF PASSED"
else
  echo "✗ AF FAILED"
fi

exit $exit_code
