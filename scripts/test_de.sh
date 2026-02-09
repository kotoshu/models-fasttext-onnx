#!/bin/bash
# Test script for DE FastText ONNX model

echo "Testing DE..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" de
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ DE PASSED"
else
  echo "✗ DE FAILED"
fi

exit $exit_code
