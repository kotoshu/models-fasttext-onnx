#!/bin/bash
# Test script for NL FastText ONNX model

echo "Testing NL..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" nl
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ NL PASSED"
else
  echo "✗ NL FAILED"
fi

exit $exit_code
