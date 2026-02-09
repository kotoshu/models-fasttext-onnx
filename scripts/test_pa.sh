#!/bin/bash
# Test script for PA FastText ONNX model

echo "Testing PA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" pa
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ PA PASSED"
else
  echo "✗ PA FAILED"
fi

exit $exit_code
