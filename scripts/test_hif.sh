#!/bin/bash
# Test script for HIF FastText ONNX model

echo "Testing HIF..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" hif
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ HIF PASSED"
else
  echo "✗ HIF FAILED"
fi

exit $exit_code
