#!/bin/bash
# Test script for CKB FastText ONNX model

echo "Testing CKB..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ckb
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ CKB PASSED"
else
  echo "✗ CKB FAILED"
fi

exit $exit_code
