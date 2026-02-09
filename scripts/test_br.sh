#!/bin/bash
# Test script for BR FastText ONNX model

echo "Testing BR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" br
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BR PASSED"
else
  echo "✗ BR FAILED"
fi

exit $exit_code
