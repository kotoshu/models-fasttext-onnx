#!/bin/bash
# Test script for MAI FastText ONNX model

echo "Testing MAI..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mai
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MAI PASSED"
else
  echo "✗ MAI FAILED"
fi

exit $exit_code
