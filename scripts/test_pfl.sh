#!/bin/bash
# Test script for PFL FastText ONNX model

echo "Testing PFL..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" pfl
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ PFL PASSED"
else
  echo "✗ PFL FAILED"
fi

exit $exit_code
