#!/bin/bash
# Test script for IS FastText ONNX model

echo "Testing IS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" is
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ IS PASSED"
else
  echo "✗ IS FAILED"
fi

exit $exit_code
