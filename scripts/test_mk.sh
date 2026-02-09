#!/bin/bash
# Test script for MK FastText ONNX model

echo "Testing MK..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mk
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MK PASSED"
else
  echo "✗ MK FAILED"
fi

exit $exit_code
