#!/bin/bash
# Test script for IT FastText ONNX model

echo "Testing IT..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" it
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ IT PASSED"
else
  echo "✗ IT FAILED"
fi

exit $exit_code
