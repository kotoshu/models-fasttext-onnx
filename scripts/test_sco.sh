#!/bin/bash
# Test script for SCO FastText ONNX model

echo "Testing SCO..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sco
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SCO PASSED"
else
  echo "✗ SCO FAILED"
fi

exit $exit_code
