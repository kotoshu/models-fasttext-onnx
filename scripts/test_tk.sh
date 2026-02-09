#!/bin/bash
# Test script for TK FastText ONNX model

echo "Testing TK..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" tk
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ TK PASSED"
else
  echo "✗ TK FAILED"
fi

exit $exit_code
