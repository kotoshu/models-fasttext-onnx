#!/bin/bash
# Test script for SCN FastText ONNX model

echo "Testing SCN..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" scn
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ SCN PASSED"
else
  echo "✗ SCN FAILED"
fi

exit $exit_code
