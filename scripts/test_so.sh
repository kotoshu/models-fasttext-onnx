#!/bin/bash
# Test script for SO FastText ONNX model

echo "Testing SO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" so
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SO PASSED"
else
  echo "✗ SO FAILED"
fi

exit $exit_code
