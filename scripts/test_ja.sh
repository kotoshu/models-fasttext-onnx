#!/bin/bash
# Test script for JA FastText ONNX model

echo "Testing JA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ja
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ JA PASSED"
else
  echo "✗ JA FAILED"
fi

exit $exit_code
