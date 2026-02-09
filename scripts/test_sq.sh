#!/bin/bash
# Test script for SQ FastText ONNX model

echo "Testing SQ..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sq
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SQ PASSED"
else
  echo "✗ SQ FAILED"
fi

exit $exit_code
