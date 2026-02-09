#!/bin/bash
# Test script for SAH FastText ONNX model

echo "Testing SAH..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" sah
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SAH PASSED"
else
  echo "✗ SAH FAILED"
fi

exit $exit_code
