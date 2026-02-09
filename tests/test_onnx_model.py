"""
FastText ONNX Model Test Framework

Tests ONNX models against YAML specifications.
Each language has its own test fixture with expected inputs and outputs.

Usage:
    python3 tests/test_onnx_model.py <language_code>

Example:
    python3 tests/test_onnx_model.py af
"""

import sys
import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime required")
    print("Install with: pip install onnxruntime")
    sys.exit(1)


class ONNXModelTester:
    """Test ONNX models against YAML specifications."""

    def __init__(self, lang: str, repo_dir: str = None):
        self.lang = lang
        self.repo_dir = Path(repo_dir) if repo_dir else Path(__file__).parent.parent
        self.spec_path = self.repo_dir / "tests" / "fixtures" / f"{lang}.yaml"
        self.model_path = self.repo_dir / "models" / lang / f"fasttext.{lang}.onnx"

    def load_spec(self) -> Dict[str, Any]:
        """Load test specification from YAML."""
        if not self.spec_path.exists():
            raise FileNotFoundError(f"Test spec not found: {self.spec_path}")

        with open(self.spec_path) as f:
            return yaml.safe_load(f)

    def load_model(self) -> ort.InferenceSession:
        """Load ONNX model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        return ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )

    def print_model_info(self, spec: Dict, sess: ort.InferenceSession):
        """Print model information."""
        metadata = spec['metadata']
        model_specs = spec['model_specifications']

        print(f"\n{metadata['language_name']} ({metadata['language_code']})")
        print("-" * 60)
        print(f"Source: {metadata['source_model']}")
        print(f"Vocabulary: {metadata['vocab_size']:,} words")
        print(f"Embedding: {metadata['embedding_dim']}D")
        print(f"ONNX: opset={metadata['onnx_opset']}, ir={metadata['onnx_ir']}")
        print()

        # Input/output specs
        input_spec = sess.get_inputs()[0]
        output_spec = sess.get_outputs()[0]
        print(f"Input:  {input_spec.name} ({input_spec.type}) {list(input_spec.shape)}")
        print(f"Output: {output_spec.name} ({output_spec.type}) {list(output_spec.shape)}")
        print()

    def validate_model_specs(self, spec: Dict, sess: ort.InferenceSession):
        """Validate model matches specifications."""
        model_specs = spec['model_specifications']

        # Check input specs
        input_spec = sess.get_inputs()[0]
        assert input_spec.name == model_specs['input']['name'], \
            f"Input name mismatch: {input_spec.name} != {model_specs['input']['name']}"
        assert list(input_spec.shape) == model_specs['input']['shape'], \
            f"Input shape mismatch: {list(input_spec.shape)} != {model_specs['input']['shape']}"

        # Check output specs
        output_spec = sess.get_outputs()[0]
        assert output_spec.name == model_specs['output']['name'], \
            f"Output name mismatch: {output_spec.name} != {model_specs['output']['name']}"
        assert list(output_spec.shape) == model_specs['output']['shape'], \
            f"Output shape mismatch: {list(output_spec.shape)} != {model_specs['output']['shape']}"

        print("  ✓ Model specifications validated")

    def run_test_case(self, sess: ort.InferenceSession, test_case: Dict, spec: Dict) -> bool:
        """Run a single test case."""
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Get input
        word_index = test_case['input']['word_index']
        print(f"\n  Test: {test_case['name']}")
        print(f"    Input: word_index = {word_index}")

        # Run inference
        embedding = sess.run([output_name], {
            input_name: np.array([word_index], dtype=np.int64)
        })[0]

        # Validate output
        expected = test_case['expected_output']

        # Check shape
        expected_shape = expected.get('shape') or expected.get('embedding_shape', [300])
        assert list(embedding.shape) == expected_shape, \
            f"    ✗ Shape mismatch: {embedding.shape} != {expected_shape}"

        # Check validation rules
        validation = spec.get('validation_rules', {})
        if validation.get('check_finite', True):
            assert np.all(np.isfinite(embedding)), "    ✗ Contains NaN or Inf"

        if validation.get('check_no_zeros', False):
            assert not np.all(embedding == 0), "    ✗ All zeros"

        # Print statistics
        mean = np.mean(embedding)
        std = np.std(embedding)
        min_val = np.min(embedding)
        max_val = np.max(embedding)

        print(f"    Output: shape={list(embedding.shape)}, mean={mean:.6f}, std={std:.6f}")
        print(f"            range=[{min_val:.4f}, {max_val:.4f}]")

        # Check sample values if provided
        if 'sample_values' in expected:
            for sample in expected['sample_values']:
                idx = sample['index']
                expected_val = sample['value']
                tolerance = sample['tolerance']
                actual_val = embedding[idx]
                diff = abs(actual_val - expected_val)
                assert diff <= tolerance, \
                    f"    ✗ Value mismatch at index {idx}: {actual_val} != {expected_val} ± {tolerance} (diff={diff})"
            print(f"    ✓ Sample values validated")

        print(f"    ✓ Test passed")
        return True

    def test_statistics(self, spec: Dict, sess: ort.InferenceSession):
        """Test vocabulary statistics if provided."""
        if 'statistics' not in spec:
            return True

        stats = spec['statistics']
        if 'vocab_sample' not in stats:
            return True

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        print("\n  Statistics validation:")

        for expected_stat in stats['vocab_sample']['expected_means']:
            idx = expected_stat['index']
            expected_mean = expected_stat['mean']
            tolerance = expected_stat['tolerance']

            embedding = sess.run([output_name], {
                input_name: np.array([idx], dtype=np.int64)
            })[0]

            actual_mean = np.mean(embedding)
            diff = abs(actual_mean - expected_mean)

            assert diff <= tolerance, \
                f"    ✗ Mean mismatch at index {idx}: {actual_mean} != {expected_mean} ± {tolerance} (diff={diff})"

            print(f"    ✓ Index {idx}: mean={actual_mean:.6f} (expected {expected_mean:.6f} ± {tolerance})")

        return True

    def run_all_tests(self) -> bool:
        """Run all tests for this language."""
        print("=" * 80)
        print(f"FASTTEXT ONNX MODEL TEST - {self.lang.upper()}")
        print("=" * 80)

        try:
            # Load specification
            spec = self.load_spec()
            print("✓ Test specification loaded")

            # Load model
            sess = self.load_model()
            print("✓ ONNX model loaded")

            # Print model info
            self.print_model_info(spec, sess)

            # Validate model specs
            self.validate_model_specs(spec, sess)

            # Run test cases
            print("\nRunning test cases:")
            for i, test_case in enumerate(spec['test_cases']):
                self.run_test_case(sess, test_case, spec)

            # Test statistics
            if 'statistics' in spec:
                self.test_statistics(spec, sess)

            print("\n" + "=" * 80)
            print(f"✓ ALL TESTS PASSED for {self.lang.upper()}")
            print("=" * 80)
            return True

        except Exception as e:
            print("\n" + "=" * 80)
            print(f"✗ TEST FAILED for {self.lang.upper()}")
            print("=" * 80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    if len(sys.argv) != 2:
        print("FastText ONNX Model Test Framework")
        print()
        print("Usage: python3 tests/test_onnx_model.py <language_code>")
        print()
        print("Examples:")
        print("  python3 tests/test_onnx_model.py af")
        print("  python3 tests/test_onnx_model.py en")
        print()
        print("Test specifications are in tests/fixtures/<lang>.yaml")
        sys.exit(1)

    lang = sys.argv[1].lower()
    tester = ONNXModelTester(lang)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
