#!/bin/bash
# Test script for WA FastText ONNX model

echo "Testing WA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" wa
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ WA PASSED"
else
  echo "✗ WA FAILED"
fi

exit $exit_code
