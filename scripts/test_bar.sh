#!/bin/bash
# Test script for BAR FastText ONNX model

echo "Testing BAR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" bar
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BAR PASSED"
else
  echo "✗ BAR FAILED"
fi

exit $exit_code
