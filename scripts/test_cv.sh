#!/bin/bash
# Test script for CV FastText ONNX model

echo "Testing CV..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" cv
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ CV PASSED"
else
  echo "✗ CV FAILED"
fi

exit $exit_code
