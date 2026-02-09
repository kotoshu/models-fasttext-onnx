#!/bin/bash
# Test script for KO FastText ONNX model

echo "Testing KO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ko
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ KO PASSED"
else
  echo "✗ KO FAILED"
fi

exit $exit_code
