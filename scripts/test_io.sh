#!/bin/bash
# Test script for IO FastText ONNX model

echo "Testing IO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" io
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ IO PASSED"
else
  echo "✗ IO FAILED"
fi

exit $exit_code
