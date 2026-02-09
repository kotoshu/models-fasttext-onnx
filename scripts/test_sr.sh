#!/bin/bash
# Test script for SR FastText ONNX model

echo "Testing SR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sr
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SR PASSED"
else
  echo "✗ SR FAILED"
fi

exit $exit_code
