#!/bin/bash
# Test script for XMF FastText ONNX model

echo "Testing XMF..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" xmf
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ XMF PASSED"
else
  echo "✗ XMF FAILED"
fi

exit $exit_code
