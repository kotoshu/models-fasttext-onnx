#!/bin/bash
# Test script for ZEA FastText ONNX model

echo "Testing ZEA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" zea
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ ZEA PASSED"
else
  echo "✗ ZEA FAILED"
fi

exit $exit_code
