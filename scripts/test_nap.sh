#!/bin/bash
# Test script for NAP FastText ONNX model

echo "Testing NAP..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" nap
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ NAP PASSED"
else
  echo "✗ NAP FAILED"
fi

exit $exit_code
