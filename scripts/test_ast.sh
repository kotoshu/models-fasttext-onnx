#!/bin/bash
# Test script for AST FastText ONNX model

echo "Testing AST..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ast
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ AST PASSED"
else
  echo "✗ AST FAILED"
fi

exit $exit_code
