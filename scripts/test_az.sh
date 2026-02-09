#!/bin/bash
# Test script for AZ FastText ONNX model

echo "Testing AZ..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" az
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ AZ PASSED"
else
  echo "✗ AZ FAILED"
fi

exit $exit_code
