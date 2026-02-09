#!/bin/bash
# Test script for UZ FastText ONNX model

echo "Testing UZ..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" uz
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ UZ PASSED"
else
  echo "✗ UZ FAILED"
fi

exit $exit_code
