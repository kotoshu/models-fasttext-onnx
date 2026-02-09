#!/bin/bash
# Test script for ARZ FastText ONNX model

echo "Testing ARZ..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" arz
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ ARZ PASSED"
else
  echo "✗ ARZ FAILED"
fi

exit $exit_code
