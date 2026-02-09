#!/bin/bash
# Test script for TT FastText ONNX model

echo "Testing TT..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" tt
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ TT PASSED"
else
  echo "✗ TT FAILED"
fi

exit $exit_code
