#!/bin/bash
# Test script for SD FastText ONNX model

echo "Testing SD..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sd
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SD PASSED"
else
  echo "✗ SD FAILED"
fi

exit $exit_code
