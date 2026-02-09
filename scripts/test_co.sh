#!/bin/bash
# Test script for CO FastText ONNX model

echo "Testing CO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" co
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ CO PASSED"
else
  echo "✗ CO FAILED"
fi

exit $exit_code
