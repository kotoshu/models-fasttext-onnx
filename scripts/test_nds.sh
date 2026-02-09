#!/bin/bash
# Test script for NDS FastText ONNX model

echo "Testing NDS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" nds
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ NDS PASSED"
else
  echo "✗ NDS FAILED"
fi

exit $exit_code
