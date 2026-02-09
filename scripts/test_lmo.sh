#!/bin/bash
# Test script for LMO FastText ONNX model

echo "Testing LMO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" lmo
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ LMO PASSED"
else
  echo "✗ LMO FAILED"
fi

exit $exit_code
