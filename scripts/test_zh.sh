#!/bin/bash
# Test script for ZH FastText ONNX model

echo "Testing ZH..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" zh
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ ZH PASSED"
else
  echo "✗ ZH FAILED"
fi

exit $exit_code
