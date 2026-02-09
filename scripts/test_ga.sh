#!/bin/bash
# Test script for GA FastText ONNX model

echo "Testing GA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ga
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ GA PASSED"
else
  echo "✗ GA FAILED"
fi

exit $exit_code
