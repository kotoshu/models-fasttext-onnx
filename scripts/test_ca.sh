#!/bin/bash
# Test script for CA FastText ONNX model

echo "Testing CA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ca
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ CA PASSED"
else
  echo "✗ CA FAILED"
fi

exit $exit_code
