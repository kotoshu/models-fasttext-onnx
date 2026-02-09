#!/bin/bash
# Test script for FRR FastText ONNX model

echo "Testing FRR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" frr
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ FRR PASSED"
else
  echo "✗ FRR FAILED"
fi

exit $exit_code
