#!/usr/bin/env python3
"""
Test script for individual FastText ONNX model.

This script performs comprehensive testing of a single ONNX model:
- Loads the model with onnxruntime
- Verifies output shape (300D)
- Tests multiple word indices
- Validates no NaN/Inf values
- Shows sample embeddings

Usage:
    python3 test_model.py <language_code>

Example:
    python3 test_model.py af
"""

import sys
import os
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime required")
    print("Install with: pip install onnxruntime")
    sys.exit(1)


def test_onnx_model(language_code):
    """Test ONNX model for a specific language."""
    model_path = f"models/{language_code}/fasttext.{language_code}.onnx"

    if not os.path.exists(model_path):
        print(f"✗ Error: Model file not found: {model_path}")
        return False

    print(f"Testing {language_code.upper()} ONNX Model")
    print("=" * 60)

    try:
        # Load model
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Verify input/output specs
        input_spec = sess.get_inputs()[0]
        output_spec = sess.get_outputs()[0]

        print(f"Model loaded successfully")
        print(f"  Input: {input_spec.name} ({input_spec.type}) shape={input_spec.shape}")
        print(f"  Output: {output_spec.name} ({output_spec.type}) shape={output_spec.shape}")

        # Test with multiple indices
        test_indices = [0, 1, 10, 100, 1000, 10000]
        print(f"\nTesting {len(test_indices)} word indices:")

        all_passed = True
        for idx in test_indices:
            embedding = sess.run([output_name], {input_name: np.array([idx], dtype=np.int64)})[0]

            # Verify shape
            if embedding.shape != (300,):
                print(f"  ✗ Index {idx}: Wrong shape {embedding.shape}, expected (300,)")
                all_passed = False
                continue

            # Verify no NaN or Inf
            if not np.all(np.isfinite(embedding)):
                print(f"  ✗ Index {idx}: Contains NaN or Inf values")
                all_passed = False
                continue

            # Calculate stats
            mean = np.mean(embedding)
            std = np.std(embedding)
            min_val = np.min(embedding)
            max_val = np.max(embedding)

            print(f"  ✓ Index {idx:5d}: shape={embedding.shape}, mean={mean:7.4f}, std={std:7.4f}, range=[{min_val:6.3f}, {max_val:6.3f}]")

        if not all_passed:
            return False

        print("\n" + "=" * 60)
        print(f"✓ ALL TESTS PASSED for {language_code.upper()}")
        print(f"  Model is functional and ready for use")
        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_model.py <language_code>")
        print("Example: python3 test_model.py af")
        sys.exit(1)

    language_code = sys.argv[1].lower()

    success = test_onnx_model(language_code)
    sys.exit(0 if success else 1)
