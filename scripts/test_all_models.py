#!/usr/bin/env python3
"""
Comprehensive test runner for all completed FastText ONNX models.

Tests all models that have:
1. An ONNX model file: models/<lang>/fasttext.<lang>.onnx
2. A test fixture: tests/fixtures/<lang>.yaml

Usage:
    python3 scripts/test_all_models.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add tests directory to path for imports
TESTS_DIR = Path(__file__).parent.parent / "tests"
sys.path.insert(0, str(TESTS_DIR))

from test_onnx_model import ONNXModelTester


def get_completed_models() -> List[str]:
    """Get list of languages with completed ONNX models and test fixtures."""
    repo_dir = Path(__file__).parent.parent
    models_dir = repo_dir / "models"
    fixtures_dir = repo_dir / "tests" / "fixtures"

    completed = []

    if not models_dir.exists():
        return completed

    for lang_dir in models_dir.iterdir():
        if not lang_dir.is_dir():
            continue

        lang = lang_dir.name
        onnx_file = lang_dir / f"fasttext.{lang}.onnx"
        fixture_file = fixtures_dir / f"{lang}.yaml"

        if onnx_file.exists() and fixture_file.exists():
            completed.append(lang)

    return sorted(completed)


def test_all_models() -> Tuple[int, int, List[str]]:
    """Test all completed models.

    Returns:
        Tuple of (passed_count, failed_count, failed_languages)
    """
    completed = get_completed_models()

    if not completed:
        print("No completed models found.")
        return 0, 0, []

    print("=" * 80)
    print(f"TESTING {len(completed)} COMPLETED MODELS")
    print("=" * 80)
    print()

    passed = 0
    failed = 0
    failed_languages = []

    for i, lang in enumerate(completed, 1):
        print(f"[{i}/{len(completed)}] Testing {lang.upper()}...")

        tester = ONNXModelTester(lang)
        success = tester.run_all_tests()

        if success:
            passed += 1
        else:
            failed += 1
            failed_languages.append(lang)

        print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total: {len(completed)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed_languages:
        print()
        print(f"Failed languages: {', '.join(failed_languages)}")

    print()
    print("Usage:")
    print("  python3 tests/test_onnx_model.py <lang>  # Test individual language")
    print("  ruby scripts/generate_test_fixtures_simple.rb  # Regenerate fixtures")

    return passed, failed, failed_languages


def main():
    passed, failed, failed_languages = test_all_models()

    # Exit with error code if any tests failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
