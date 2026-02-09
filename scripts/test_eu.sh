#!/bin/bash
# Test script for EU FastText ONNX model

echo "Testing EU..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" eu
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ EU PASSED"
else
  echo "✗ EU FAILED"
fi

exit $exit_code
