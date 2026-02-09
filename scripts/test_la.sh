#!/bin/bash
# Test script for LA FastText ONNX model

echo "Testing LA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" la
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ LA PASSED"
else
  echo "✗ LA FAILED"
fi

exit $exit_code
