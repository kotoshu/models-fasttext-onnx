#!/bin/bash
# Test script for ES FastText ONNX model

echo "Testing ES..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" es
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ ES PASSED"
else
  echo "✗ ES FAILED"
fi

exit $exit_code
