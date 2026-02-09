#!/bin/bash
# Test script for MS FastText ONNX model

echo "Testing MS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ms
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MS PASSED"
else
  echo "✗ MS FAILED"
fi

exit $exit_code
