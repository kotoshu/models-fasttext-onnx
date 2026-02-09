#!/bin/bash
# Test script for BA FastText ONNX model

echo "Testing BA..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ba
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BA PASSED"
else
  echo "✗ BA FAILED"
fi

exit $exit_code
