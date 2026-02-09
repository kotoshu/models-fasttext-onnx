#!/bin/bash
# Test script for NN FastText ONNX model

echo "Testing NN..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" nn
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ NN PASSED"
else
  echo "✗ NN FAILED"
fi

exit $exit_code
