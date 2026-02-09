#!/bin/bash
# Test script for BPY FastText ONNX model

echo "Testing BPY..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" bpy
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ BPY PASSED"
else
  echo "✗ BPY FAILED"
fi

exit $exit_code
