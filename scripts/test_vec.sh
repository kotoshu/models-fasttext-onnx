#!/bin/bash
# Test script for VEC FastText ONNX model

echo "Testing VEC..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" vec
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ VEC PASSED"
else
  echo "✗ VEC FAILED"
fi

exit $exit_code
