#!/bin/bash
# Test script for PAM FastText ONNX model

echo "Testing PAM..."
python3 "/Users/mulgogi/src/kotoshu/models-fasttext-onnx/scripts/test_model.py" pam
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "✓ PAM PASSED"
else
  echo "✗ PAM FAILED"
fi

exit $exit_code
