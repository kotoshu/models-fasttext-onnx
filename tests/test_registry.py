"""Registry validator tests against the self-contained fixture."""
import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_registry as vr  # noqa: E402

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
    def _bcp47_fixture_root(self, lang: str):
        """A temp fixture whose first model entry carries `lang` - the
        minimal probe for the plan-19 BCP-47 widening. Returns the temp
        repo root for run_validator."""
        import shutil, tempfile
        reg = json.loads((FIXTURE / "registry.json").read_text(encoding="utf-8"))
        for rid, entry in list(reg["resources"].items()):
            if entry.get("type") == "model":
                old = entry["language"]
                entry["language"] = lang
                rid_new = rid.replace(f"/{old}/", f"/{lang}/")
                entry["urls"]["mirror"] = (
                    "https://media.githubusercontent.com/media/kotoshu/models-fasttext-onnx"
                    f"/main/models/{lang}/fasttext.{lang}.onnx"
                )
                reg["resources"][rid_new] = reg["resources"].pop(rid)
                manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
                mres = manifest["resources"]
                mkey = next((k for k in mres if k.startswith(f"models/{old}/") and k.endswith(".onnx")), None)
                if mkey is None:
                    self.fail("fixture manifest lacks the model artifact")
                new_mkey = f"models/{lang}/fasttext.{lang}.onnx"
                entry_copy = dict(mres[mkey]); entry_copy["language"] = lang
                mres[new_mkey] = entry_copy
                break
        else:
            self.fail("fixture has no model entry")
        tmp = Path(tempfile.mkdtemp(prefix="bcp47-fixture-"))
        shutil.copytree(FIXTURE, tmp, dirs_exist_ok=True)
        (tmp / "registry.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1))
        (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp

    def test_bcp47_variant_codes_accepted(self):
        # plan 19: zh-Hans-CN / zh-Hant-TW / zh-Hant-HK are the owner-approved
        # variant codes; 3-letter ISO bases (the old nds rejection) pass too.
        for lang in ("zh-Hans-CN", "zh-Hant-TW", "zh-Hant-HK", "nds"):
            root = self._bcp47_fixture_root(lang)
            result = run_validator(root)
            self.assertEqual(
                result.returncode, 0,
                f"validator rejected BCP-47 language {lang}:\n{result.stdout}\n{result.stderr}",
            )

    def test_bcp47_canonical_casing_enforced(self):
        # Non-canonical casing (zh-hans-cn) is REJECTED - the registry
        # carries exactly one spelling per variant.
        root = self._bcp47_fixture_root("zh-hans-cn")
        result = run_validator(root)
        self.assertNotEqual(result.returncode, 0, "validator accepted lowercase subtags")

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

    def run_generator(self, repo_root, tag=None):
        cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "build_registry.py"),
               "--repo-root", str(repo_root)]
        if tag:
            cmd += ["--tag", tag]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_typo_released_pair_generates_and_validates(self):
        # Plan 131: with the descriptor's release_tag set, the generator
        # emits BOTH the primary and the vocab URL at the release-tag
        # convention, and the validator accepts the full released state.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            descriptor_path = copy / "models" / "typo" / "typo.json"
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            descriptor["release_tag"] = "v1.6.1"
            descriptor_path.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")

            result = self.run_generator(copy, tag="v1.6.1")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            registry = json.loads((copy / "registry.json").read_text(encoding="utf-8"))
            typo = registry["resources"]["kotoshu://models/typo/typo-biencoder"]
            base = "https://github.com/kotoshu/models-fasttext-onnx/releases/download/v1.6.1"
            self.assertEqual(typo["urls"]["primary"], f"{base}/typo.biencoder.onnx")
            self.assertEqual(typo["vocab_url"], f"{base}/typo.biencoder.vocab.json")

            validation = run_validator(copy)
            self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)

    def seed_matrix_descriptor(self, copy, paired_sha=None):
        manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
        full_sha = manifest["resources"]["models/de/fasttext.de.onnx"]["sha256"]
        paired = paired_sha or full_sha
        (copy / "models" / "de" / "typo-matrix.json").write_text(json.dumps({
            "plan": "14 test", "kind": "matrix", "vocab_size": 10, "dims": 256,
            "bytes": 2616, "sha256": "f" * 64,
            "paired_vocab": f"kotoshu://models/de/full @ sha256 {paired}",
            "release_tag": None,
        }, indent=2), encoding="utf-8")
        return full_sha

    def test_matrix_pairing_travels_and_cross_checks(self):
        # Plan 14: the pairing sha rides the entry, and the validator
        # compares it against the full tier's ground truth — a rebuilt
        # tier without a rebuilt matrix fails loudly.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            full_sha = self.seed_matrix_descriptor(copy)

            result = self.run_generator(copy)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            registry = json.loads((copy / "registry.json").read_text(encoding="utf-8"))
            entry = registry["resources"]["kotoshu://models/de/typo-matrix"]
            self.assertEqual(entry["paired_vocab_sha256"], full_sha)

            validation = run_validator(copy)
            self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)

    def test_matrix_pairing_mismatch_fails_with_rebuild_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            self.seed_matrix_descriptor(copy, paired_sha="e" * 64)

            result = self.run_generator(copy)
            self.assertEqual(result.returncode, 0)
            validation = run_validator(copy)
            self.assertNotEqual(validation.returncode, 0)
            self.assertIn("rebuild the matrix", validation.stdout + validation.stderr)

    def test_matrix_pairing_field_missing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            self.seed_matrix_descriptor(copy)
            self.run_generator(copy)
            registry_path = copy / "registry.json"
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            del data["resources"]["kotoshu://models/de/typo-matrix"]["paired_vocab_sha256"]
            registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            result = run_validator(copy)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("paired_vocab_sha256 missing or malformed",
                          result.stdout + result.stderr)

    def test_typo_prerelease_state_generates_nulls(self):
        # Pre-release (descriptor release_tag null): both URLs stay null
        # — the frozen v1.6.0 state regenerates unchanged.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            result = self.run_generator(copy)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            registry = json.loads((copy / "registry.json").read_text(encoding="utf-8"))
            typo = registry["resources"]["kotoshu://models/typo/typo-biencoder"]
            self.assertIsNone(typo["urls"]["primary"])
            self.assertIsNone(typo["vocab_url"])
            validation = run_validator(copy)
            self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)

    def test_typo_primary_without_vocab_fails(self):
        # The pair travels together: a released primary without its
        # vocab sibling is a dead half-pair.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            registry_path = copy / "registry.json"
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            tag = data.get("release_tag") or "v1.6.0"
            data["release_tag"] = tag
            typo = data["resources"]["kotoshu://models/typo/typo-biencoder"]
            base = f"https://github.com/kotoshu/models-fasttext-onnx/releases/download/{tag}"
            typo["urls"]["primary"] = f"{base}/typo.biencoder.onnx"
            typo["vocab_url"] = None
            registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            result = run_validator(copy)
            self.assertIn("must either both be null",
                          result.stdout + result.stderr)

    def test_typo_primary_off_convention_fails(self):
        # A primary that does not match the registry's release tag is
        # the dead-URL failure mode the validator has always rejected.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "registry_fixture"
            shutil.copytree(FIXTURE, copy)
            registry_path = copy / "registry.json"
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            data["release_tag"] = "v1.6.0"
            typo = data["resources"]["kotoshu://models/typo/typo-biencoder"]
            base = "https://github.com/kotoshu/models-fasttext-onnx/releases/download/v0.0.1-other"
            typo["urls"]["primary"] = f"{base}/typo.biencoder.onnx"
            typo["vocab_url"] = f"{base}/typo.biencoder.vocab.json"
            registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            result = run_validator(copy)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SAME semver release tag", result.stdout + result.stderr)

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


class CheckUrlsLiveTest(unittest.TestCase):
    """Plan 10: live URL probing against a real local HTTP server.

    The registry's mirror convention assumes LFS-tracked artifacts; a
    plain-git blob 404s on the media host and construction-only checks
    cannot see it. These tests exercise probe_url/check_urls_live with
    real bytes over real HTTP — hermetic, no internet.
    """

    def serve(self, files):
        root = Path(tempfile.mkdtemp())
        for name, blob in files.items():
            (root / name).write_bytes(blob)
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(root))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def resource(self, mirror=None, primary=None, vocab_url=None, size_bytes=None):
        return {
            "type": "model", "language": "en",
            "urls": {"primary": primary, "mirror": mirror},
            "vocab_url": vocab_url, "size_bytes": size_bytes,
        }

    def test_reachable_url_with_matching_size_passes(self):
        base = self.serve({"a.onnx": b"x"})
        resources = {"kotoshu://models/en/full":
                     self.resource(mirror=f"{base}/a.onnx", size_bytes=1)}
        errors = []
        self.assertEqual(vr.check_urls_live(resources, errors), 1)
        self.assertEqual(errors, [])

    def test_null_urls_are_not_probed(self):
        resources = {"kotoshu://models/en/full": self.resource(size_bytes=1)}
        self.assertEqual(vr.check_urls_live(resources, []), 0)

    def test_unreachable_mirror_fails(self):
        base = self.serve({"a.onnx": b"x"})
        resources = {"kotoshu://models/en/full":
                     self.resource(mirror=f"{base}/missing.onnx", size_bytes=1)}
        errors = []
        vr.check_urls_live(resources, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("unreachable (HTTP 404)", errors[0])
        self.assertIn("mirror", errors[0])

    def test_served_size_mismatching_declaration_fails(self):
        base = self.serve({"a.onnx": b"xy"})
        resources = {"kotoshu://models/en/full":
                     self.resource(mirror=f"{base}/a.onnx", size_bytes=1)}
        errors = []
        vr.check_urls_live(resources, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("serves 2 bytes, registry declares 1", errors[0])

    def test_vocab_url_is_reachability_only(self):
        base = self.serve({"a.onnx": b"x", "a.vocab.json": b'{"w":1}'})
        resources = {"kotoshu://models/en/full":
                     self.resource(mirror=f"{base}/a.onnx",
                                   vocab_url=f"{base}/a.vocab.json", size_bytes=1)}
        errors = []
        # 2 probes (mirror + vocab); the 3-byte vocab body never trips the
        # size gate — its authoritative size lives in ground truth.
        self.assertEqual(vr.check_urls_live(resources, errors), 2)
        self.assertEqual(errors, [])

    def test_mirror_probe_url_rewrites_main_to_the_validating_ref(self):
        mirror = f"{vr.MEDIA_URL}/main/models/en/typo.matrix.en.ktm1"
        self.assertEqual(vr.mirror_probe_url(mirror, "main"), mirror)
        self.assertEqual(vr.mirror_probe_url(mirror, "plan-12-matrices"),
                         f"{vr.MEDIA_URL}/plan-12-matrices/models/en/typo.matrix.en.ktm1")
        self.assertEqual(vr.mirror_probe_url(mirror, None), mirror)
        release = "https://github.com/kotoshu/models-fasttext-onnx/releases/download/v1.7.0/x.onnx"
        self.assertEqual(vr.mirror_probe_url(release, "plan-12-matrices"), release)

    def test_check_urls_live_probes_the_ref_rewritten_url(self):
        # The rewrite decides WHICH host path gets probed: a branch-added
        # artifact 404s on main until merge. The local server stands in
        # for the media host by serving the same path shape, reachable
        # only under the branch segment — proving the probe followed the
        # ref, not the /main/ the registry names.
        branch = "plan-12-matrices"
        root = Path(tempfile.mkdtemp())
        (root / branch / "models" / "de").mkdir(parents=True)
        (root / branch / "models" / "de" / "typo.matrix.de.ktm1").write_bytes(b"x")
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(root))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        self.base_url = f"http://127.0.0.1:{server.server_address[1]}"

        resources = {"kotoshu://models/de/typo-matrix":
                     self.resource(mirror=f"{self.base_url}/main/models/de/typo.matrix.de.ktm1",
                                   size_bytes=1)}
        real_media_url = vr.MEDIA_URL
        vr.MEDIA_URL = self.base_url
        try:
            errors = []
            vr.check_urls_live(resources, errors, ref=branch)
        finally:
            vr.MEDIA_URL = real_media_url
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
