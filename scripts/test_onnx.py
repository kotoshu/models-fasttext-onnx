#!/usr/bin/env python3
"""
Test ONNX model for a specific language.

Verifies:
1. Model loads with onnxruntime
2. Returns 300-dimensional embeddings
3. Multiple queries work
4. Valid floating-point values (no NaN/Inf)

Usage:
    python3 test_onnx.py <language_code>

Example:
    python3 test_onnx.py en
"""

import sys
import os
import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Error: onnx and onnxruntime required")
    print("Install with: pip install onnx onnxruntime")
    sys.exit(1)


def test_onnx_model(language_code):
    """Test ONNX model for a language."""
    model_path = f"models/{language_code}/fasttext.{language_code}.onnx"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    print(f"Testing {language_code.upper()} ONNX Model")
    print("=" * 60)

    # Load model with onnx
    model = onnx.load(model_path)
    print(f"ONNX IR version: {model.ir_version}")
    print(f"ONNX opset version: {model.opset_import[0].version}")

    # Create inference session
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Get input/output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    print(f"\nModel Input: {input_name}")
    print(f"Model Output: {output_name}")

    # Test with multiple indices
    test_indices = [0, 1, 100, 1000]

    print("\nTesting embeddings:")
    for idx in test_indices:
        embedding = sess.run([output_name], {input_name: np.array([idx], dtype=np.int64)})[0][0]

        # Verify shape
        if embedding.shape != (300,):
            print(f"  ✗ Index {idx}: Wrong shape {embedding.shape}")
            return False

        # Verify no NaN or Inf
        if not np.all(np.isfinite(embedding)):
            print(f"  ✗ Index {idx}: Contains NaN or Inf")
            return False

        # Calculate stats
        mean = np.mean(embedding)
        std = np.std(embedding)
        min_val = np.min(embedding)
        max_val = np.max(embedding)

        print(f"  ✓ Index {idx}: shape={embedding.shape}, mean={mean:.6f}, std={std:.6f}")
        print(f"    Range: [{min_val:.4f}, {max_val:.4f}]")

    print("\n" + "=" * 60)
    print(f"✓ ALL TESTS PASSED for {language_code.upper()}")
    print("\nModel is functional and ready for use!")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_onnx.py <language_code>")
        print("Example: python3 test_onnx.py en")
        sys.exit(1)

    language_code = sys.argv[1].lower()

    valid_languages = ['de', 'en', 'es', 'fr', 'pt', 'ru']
    if language_code not in valid_languages:
        print(f"Error: Invalid language code '{language_code}'")
        print(f"Valid languages: {', '.join(valid_languages)}")
        sys.exit(1)

    success = test_onnx_model(language_code)
    sys.exit(0 if success else 1)
