#!/bin/bash
# Test script for EO FastText ONNX model

echo "Testing EO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" eo
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ EO PASSED"
else
  echo "✗ EO FAILED"
fi

exit $exit_code
