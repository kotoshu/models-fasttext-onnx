#!/bin/bash
# Test script for IA FastText ONNX model

echo "Testing IA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ia
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ IA PASSED"
else
  echo "✗ IA FAILED"
fi

exit $exit_code
