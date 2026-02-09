#!/bin/bash
# Test script for MRJ FastText ONNX model

echo "Testing MRJ..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" mrj
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ MRJ PASSED"
else
  echo "✗ MRJ FAILED"
fi

exit $exit_code
