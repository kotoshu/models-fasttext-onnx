#!/bin/bash
# Test script for MWL FastText ONNX model

echo "Testing MWL..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mwl
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MWL PASSED"
else
  echo "✗ MWL FAILED"
fi

exit $exit_code
