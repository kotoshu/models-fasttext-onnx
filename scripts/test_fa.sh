#!/bin/bash
# Test script for FA FastText ONNX model

echo "Testing FA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" fa
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ FA PASSED"
else
  echo "✗ FA FAILED"
fi

exit $exit_code
