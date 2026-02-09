#!/bin/bash
# Test script for SC FastText ONNX model

echo "Testing SC..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sc
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SC PASSED"
else
  echo "✗ SC FAILED"
fi

exit $exit_code
