#!/bin/bash
# Test script for AZB FastText ONNX model

echo "Testing AZB..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" azb
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ AZB PASSED"
else
  echo "✗ AZB FAILED"
fi

exit $exit_code
