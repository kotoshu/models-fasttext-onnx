#!/bin/bash
# Test script for MHR FastText ONNX model

echo "Testing MHR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mhr
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MHR PASSED"
else
  echo "✗ MHR FAILED"
fi

exit $exit_code
