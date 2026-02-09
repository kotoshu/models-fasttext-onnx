#!/bin/bash
# Test script for PT FastText ONNX model

echo "Testing PT..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" pt
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ PT PASSED"
else
  echo "✗ PT FAILED"
fi

exit $exit_code
