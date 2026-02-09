#!/bin/bash
# Test script for YO FastText ONNX model

echo "Testing YO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" yo
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ YO PASSED"
else
  echo "✗ YO FAILED"
fi

exit $exit_code
