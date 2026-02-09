#!/bin/bash
# Test script for TE FastText ONNX model

echo "Testing TE..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" te
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ TE PASSED"
else
  echo "✗ TE FAILED"
fi

exit $exit_code
