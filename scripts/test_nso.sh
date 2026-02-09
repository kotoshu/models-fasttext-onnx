#!/bin/bash
# Test script for NSO FastText ONNX model

echo "Testing NSO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" nso
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ NSO PASSED"
else
  echo "✗ NSO FAILED"
fi

exit $exit_code
