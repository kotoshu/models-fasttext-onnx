#!/bin/bash
# Test script for HE FastText ONNX model

echo "Testing HE..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" he
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ HE PASSED"
else
  echo "✗ HE FAILED"
fi

exit $exit_code
