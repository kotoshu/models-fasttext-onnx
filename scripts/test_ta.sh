#!/bin/bash
# Test script for TA FastText ONNX model

echo "Testing TA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ta
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ TA PASSED"
else
  echo "✗ TA FAILED"
fi

exit $exit_code
