#!/bin/bash
# Test script for UK FastText ONNX model

echo "Testing UK..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" uk
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ UK PASSED"
else
  echo "✗ UK FAILED"
fi

exit $exit_code
