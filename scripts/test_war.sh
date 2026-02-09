#!/bin/bash
# Test script for WAR FastText ONNX model

echo "Testing WAR..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" war
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ WAR PASSED"
else
  echo "✗ WAR FAILED"
fi

exit $exit_code
