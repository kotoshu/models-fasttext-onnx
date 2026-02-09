#!/bin/bash
# Test script for TR FastText ONNX model

echo "Testing TR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" tr
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ TR PASSED"
else
  echo "✗ TR FAILED"
fi

exit $exit_code
