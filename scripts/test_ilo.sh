#!/bin/bash
# Test script for ILO FastText ONNX model

echo "Testing ILO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ilo
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ ILO PASSED"
else
  echo "✗ ILO FAILED"
fi

exit $exit_code
