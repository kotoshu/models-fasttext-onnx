#!/bin/bash
# Test script for AS FastText ONNX model

echo "Testing AS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" as
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ AS PASSED"
else
  echo "✗ AS FAILED"
fi

exit $exit_code
