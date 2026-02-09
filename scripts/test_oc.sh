#!/bin/bash
# Test script for OC FastText ONNX model

echo "Testing OC..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" oc
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ OC PASSED"
else
  echo "✗ OC FAILED"
fi

exit $exit_code
