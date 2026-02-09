#!/bin/bash
# Test script for PS FastText ONNX model

echo "Testing PS..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ps
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ PS PASSED"
else
  echo "✗ PS FAILED"
fi

exit $exit_code
