#!/bin/bash
# Test script for DA FastText ONNX model

echo "Testing DA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" da
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ DA PASSED"
else
  echo "✗ DA FAILED"
fi

exit $exit_code
