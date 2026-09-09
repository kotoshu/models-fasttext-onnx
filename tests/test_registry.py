"""Registry validator tests against the self-contained fixture."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_registry.py"
FIXTURE = REPO_ROOT / "tests" / "registry_fixture"
SCHEMA = REPO_ROOT / "schemas" / "registry.schema.json"


def run_validator(repo_root, check_files=False):
    cmd = [
        sys.executable, str(VALIDATOR),
        "--repo-root", str(repo_root),
        "--schema", str(SCHEMA),
        "--registry", str(Path(repo_root) / "registry.json"),
    ]
    if check_files:
        cmd.append("--check-files")
    return subprocess.run(cmd, capture_output=True, text=True)


class ValidateRegistryTest(unittest.TestCase):
    def test_fixture_passes(self):
        result = run_validator(FIXTURE)
        self.assertEqual(
            result.returncode, 0,
            f"validator failed:\n{result.stdout}\n{result.stderr}",
        )

    def test_fixture_pack_passes_file_checks(self):
        # The fixture pack (plan 113) is structurally verified end to
        # end: framing walk, per-section footers, declared offsets.
        result = run_validator(FIXTURE, check_files=True)
        self.assertEqual(
            result.returncode, 0,
            f"validator failed:\n{result.stdout}\n{result.stderr}",
        )

    def test_corrupted_sha256_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            registry_path = copy / "registry.json"
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            first_id = next(iter(data["resources"]))
            resource = data["resources"][first_id]
            sha = resource["sha256"]
            prefix = "ab" if sha[:2] != "ab" else "ba"
            resource["sha256"] = prefix + sha[2:]
            registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            result = run_validator(copy)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sha256", result.stdout + result.stderr)

    def test_pack_descriptor_drift_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            descriptor_path = copy / "packs" / "packs.json"
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            descriptor["packs"]["en"]["dictionary_pin"] = "f" * 40
            descriptor_path.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
            result = run_validator(copy)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dictionary_pin", result.stdout + result.stderr)

    def test_typo_descriptor_drift_fails(self):
        # Plan 115: models/typo/typo.json is the typo-bi-encoder ground
        # truth; sha drift between descriptor and registry must fail.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            descriptor_path = copy / "models" / "typo" / "typo.json"
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            sha = descriptor["sha256"]
            prefix = "ab" if sha[:2] != "ab" else "ba"
            descriptor["sha256"] = prefix + sha[2:]
            descriptor_path.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
            result = run_validator(copy)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("models/typo/typo.json", result.stdout + result.stderr)

    def test_typo_primary_url_set_fails(self):
        # Plan 115 keeps the typo bi-encoder media-host only (the plan-113
        # additive template): a primary URL before a release carries the
        # assets is a registry error (dead-URL failure mode).
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            registry_path = copy / "registry.json"
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            data["resources"]["kotoshu://models/typo/typo-biencoder"]["urls"]["primary"] = (
                "https://github.com/kotoshu/models-fasttext-onnx/releases/download/v9.9.9/typo.biencoder.onnx"
            )
            registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            result = run_validator(copy)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("typo primary", result.stdout + result.stderr)

    def test_tampered_pack_file_fails_file_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            pack_path = copy / "packs" / "en-0.0.0-dev.bin"
            data = bytearray(pack_path.read_bytes())
            data[64] ^= 0xFF  # inside a section payload
            pack_path.write_bytes(bytes(data))
            result = run_validator(copy, check_files=True)
            self.assertNotEqual(result.returncode, 0)
            # The tamper trips either the whole-file sha256 gate or the
            # framing walk (both are pack checks).
            self.assertIn("pack", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
