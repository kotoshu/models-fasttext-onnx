#!/bin/bash
# Test script for EN FastText ONNX model

echo "Testing EN..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" en
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ EN PASSED"
else
  echo "✗ EN FAILED"
fi

exit $exit_code
