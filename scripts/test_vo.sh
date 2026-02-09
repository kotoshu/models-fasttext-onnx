#!/bin/bash
# Test script for VO FastText ONNX model

echo "Testing VO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" vo
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ VO PASSED"
else
  echo "✗ VO FAILED"
fi

exit $exit_code
