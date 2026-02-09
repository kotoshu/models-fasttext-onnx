#!/bin/bash
# Test script for SV FastText ONNX model

echo "Testing SV..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sv
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SV PASSED"
else
  echo "✗ SV FAILED"
fi

exit $exit_code
