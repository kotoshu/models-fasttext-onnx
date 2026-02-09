#!/bin/bash
# Test script for NE FastText ONNX model

echo "Testing NE..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ne
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ NE PASSED"
else
  echo "✗ NE FAILED"
fi

exit $exit_code
