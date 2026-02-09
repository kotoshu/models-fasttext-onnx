#!/bin/bash
# Test script for HT FastText ONNX model

echo "Testing HT..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ht
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ HT PASSED"
else
  echo "✗ HT FAILED"
fi

exit $exit_code
