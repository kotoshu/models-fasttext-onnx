#!/bin/bash
# Test script for ID FastText ONNX model

echo "Testing ID..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" id
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ ID PASSED"
else
  echo "✗ ID FAILED"
fi

exit $exit_code
