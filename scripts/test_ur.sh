#!/bin/bash
# Test script for UR FastText ONNX model

echo "Testing UR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ur
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ UR PASSED"
else
  echo "✗ UR FAILED"
fi

exit $exit_code
