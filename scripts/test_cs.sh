#!/bin/bash
# Test script for CS FastText ONNX model

echo "Testing CS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" cs
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ CS PASSED"
else
  echo "✗ CS FAILED"
fi

exit $exit_code
