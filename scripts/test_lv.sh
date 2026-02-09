#!/bin/bash
# Test script for LV FastText ONNX model

echo "Testing LV..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" lv
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ LV PASSED"
else
  echo "✗ LV FAILED"
fi

exit $exit_code
