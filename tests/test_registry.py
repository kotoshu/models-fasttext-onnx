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


def run_validator(repo_root):
    cmd = [
        sys.executable, str(VALIDATOR),
        "--repo-root", str(repo_root),
        "--schema", str(SCHEMA),
        "--registry", str(Path(repo_root) / "registry.json"),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


class ValidateRegistryTest(unittest.TestCase):
    def test_fixture_passes(self):
        result = run_validator(FIXTURE)
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


if __name__ == "__main__":
    unittest.main()
