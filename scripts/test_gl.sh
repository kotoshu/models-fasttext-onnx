#!/bin/bash
# Test script for GL FastText ONNX model

echo "Testing GL..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" gl
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ GL PASSED"
else
  echo "✗ GL FAILED"
fi

exit $exit_code
