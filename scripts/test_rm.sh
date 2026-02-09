#!/bin/bash
# Test script for RM FastText ONNX model

echo "Testing RM..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" rm
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ RM PASSED"
else
  echo "✗ RM FAILED"
fi

exit $exit_code
