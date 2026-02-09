#!/bin/bash
# Test script for ML FastText ONNX model

echo "Testing ML..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" ml
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ ML PASSED"
else
  echo "✗ ML FAILED"
fi

exit $exit_code
