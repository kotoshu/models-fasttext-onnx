#!/bin/bash
# Test script for PMS FastText ONNX model

echo "Testing PMS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" pms
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ PMS PASSED"
else
  echo "✗ PMS FAILED"
fi

exit $exit_code
