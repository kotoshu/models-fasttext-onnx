#!/bin/bash
# Test script for NO FastText ONNX model

echo "Testing NO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" no
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ NO PASSED"
else
  echo "✗ NO FAILED"
fi

exit $exit_code
