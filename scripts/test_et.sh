#!/bin/bash
# Test script for ET FastText ONNX model

echo "Testing ET..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" et
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ ET PASSED"
else
  echo "✗ ET FAILED"
fi

exit $exit_code
