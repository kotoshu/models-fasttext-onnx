#!/bin/bash
# Test script for GV FastText ONNX model

echo "Testing GV..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" gv
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ GV PASSED"
else
  echo "✗ GV FAILED"
fi

exit $exit_code
