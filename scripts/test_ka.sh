#!/bin/bash
# Test script for KA FastText ONNX model

echo "Testing KA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ka
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ KA PASSED"
else
  echo "✗ KA FAILED"
fi

exit $exit_code
