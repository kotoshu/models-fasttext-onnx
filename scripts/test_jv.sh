#!/bin/bash
# Test script for JV FastText ONNX model

echo "Testing JV..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" jv
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ JV PASSED"
else
  echo "✗ JV FAILED"
fi

exit $exit_code
