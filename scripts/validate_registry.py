#!/usr/bin/env python3
"""Validate registry.json.

Always checks schema conformance (schemas/registry.schema.json), id
uniqueness, URL construction, and consistency with the ground-truth
sources (manifest.json for the full tier, models/<lang>/tiers.json for
derived tiers). --check-files additionally hashes local model and vocab
files when present; absent files are warnings (e.g. CI before a tier
build), mismatches are failures. Exits nonzero on any failure.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

REPO_URL = "https://github.com/kotoshu/models-fasttext-onnx"
MEDIA_URL = "https://media.githubusercontent.com/media/kotoshu/models-fasttext-onnx"
READ_CHUNK = 1 << 20

format_checker = FormatChecker()


@format_checker.checks("date-time", raises=(ValueError,))
def _is_iso8601(value):
    # jsonschema needs rfc3339-validator for date-time; keep the check
    # dependency-free by parsing ourselves (Z suffix needs normalizing
    # before Python 3.11).
    if not isinstance(value, str):
        return True
    datetime.fromisoformat(value.replace("Z", "+00:00"))
    return True


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256_and_size(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def asset_stems(lang, tier_name):
    stem = f"fasttext.{lang}" if tier_name == "full" else f"fasttext.{lang}.{tier_name}"
    return f"{stem}.onnx", f"{stem}.vocab.json"


def check_urls(resource, resource_id, registry, errors):
    lang = resource["language"]
    tier_name = resource["tier"]["name"]
    onnx_name, vocab_name = asset_stems(lang, tier_name)
    tag = registry["release_tag"]

    # Only the full tier is in git (LFS -> media host; the raw host
    # serves pointer stubs). Tier binaries are release assets only.
    if tier_name == "full":
        expected_mirror = f"{MEDIA_URL}/main/models/{lang}/{onnx_name}"
        if resource["urls"]["mirror"] != expected_mirror:
            errors.append(f"{resource_id}: mirror URL expected {expected_mirror}")
    elif resource["urls"]["mirror"] is not None:
        errors.append(f"{resource_id}: tier mirror must be null (release assets only)")

    if tag is None:
        if resource["urls"]["primary"] is not None or resource["vocab_url"] is not None:
            errors.append(f"{resource_id}: primary/vocab URLs set but release_tag is null")
    else:
        expected_primary = f"{REPO_URL}/releases/download/{tag}/{onnx_name}"
        expected_vocab = f"{REPO_URL}/releases/download/{tag}/{vocab_name}"
        if resource["urls"]["primary"] != expected_primary:
            errors.append(f"{resource_id}: primary URL expected {expected_primary}")
        if resource["vocab_url"] != expected_vocab:
            errors.append(f"{resource_id}: vocab_url expected {expected_vocab}")


def check_ground_truth(resource, resource_id, root, manifest, errors):
    lang = resource["language"]
    tier_name = resource["tier"]["name"]

    if tier_name == "full":
        entry = manifest["resources"].get(f"models/{lang}/fasttext.{lang}.onnx")
        if entry is None:
            errors.append(f"{resource_id}: no manifest.json entry for models/{lang}/fasttext.{lang}.onnx")
            return None
        if resource["sha256"] != entry["sha256"] or resource["size_bytes"] != entry["size"]:
            errors.append(f"{resource_id}: sha256/size drift vs manifest.json")
        vocab_entry = manifest["resources"].get(f"models/{lang}/fasttext.{lang}.vocab.json")
        return None if vocab_entry is None else (vocab_entry["sha256"], vocab_entry["size"])

    tiers_path = root / "models" / lang / "tiers.json"
    try:
        tiers = load_json(tiers_path)
        t = tiers["tiers"][tier_name]
    except (OSError, KeyError):
        errors.append(f"{resource_id}: no models/{lang}/tiers.json entry for tier {tier_name!r}")
        return None
    if resource["sha256"] != t["sha256"] or resource["size_bytes"] != t["bytes"]:
        errors.append(f"{resource_id}: sha256/size drift vs models/{lang}/tiers.json")
    return (t["vocab_sha256"], t["vocab_bytes"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", help="path to registry.json (default: <repo-root>/registry.json)")
    ap.add_argument("--repo-root", default=".", help="repository root (default: current directory)")
    ap.add_argument("--schema", help="path to registry.schema.json (default: <repo-root>/schemas/registry.schema.json)")
    ap.add_argument("--check-files", action="store_true",
                    help="hash local model/vocab files present on disk (absent files are warnings)")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    registry_path = Path(args.registry) if args.registry else root / "registry.json"

    errors = []
    warnings = []
    try:
        registry = load_json(registry_path)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read {registry_path}: {exc}")

    schema_path = Path(args.schema) if args.schema else root / "schemas" / "registry.schema.json"
    try:
        schema = load_json(schema_path)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read {schema_path}: {exc}")

    validator = Draft202012Validator(schema, format_checker=format_checker)
    for error in sorted(validator.iter_errors(registry), key=lambda e: list(e.absolute_path)):
        errors.append(f"schema: {'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}")

    resources = registry.get("resources", {})
    ids = list(resources)
    # JSON objects cannot carry duplicate keys post-parse; kept as an
    # explicit invariant so a future list-based shape cannot regress it.
    if len(set(ids)) != len(ids):
        errors.append("duplicate resource ids present")

    manifest = load_json(root / "manifest.json") if (root / "manifest.json").exists() else None
    if manifest is None:
        errors.append("manifest.json not found; ground-truth comparison skipped")

    # Deep checks assume the schema shape; skip them when it already failed.
    for resource_id, resource in (resources.items() if not errors else []):
        check_urls(resource, resource_id, registry, errors)
        if manifest is not None:
            vocab_truth = check_ground_truth(resource, resource_id, root, manifest, errors)
        else:
            vocab_truth = None

        if not args.check_files:
            continue
        lang = resource["language"]
        tier_name = resource["tier"]["name"]
        onnx_name, vocab_name = asset_stems(lang, tier_name)

        onnx_path = root / "models" / lang / onnx_name
        if onnx_path.exists():
            sha, size = file_sha256_and_size(onnx_path)
            if sha != resource["sha256"] or size != resource["size_bytes"]:
                errors.append(f"{resource_id}: local file {onnx_path} does not match registry sha256/size")
        else:
            warnings.append(f"{resource_id}: {onnx_path} absent locally, file check skipped")

        vocab_path = root / "models" / lang / vocab_name
        if vocab_path.exists():
            if vocab_truth is None:
                warnings.append(f"{resource_id}: {vocab_path} exists but no vocab ground truth, check skipped")
            else:
                sha, size = file_sha256_and_size(vocab_path)
                if sha != vocab_truth[0] or size != vocab_truth[1]:
                    errors.append(f"{resource_id}: local vocab {vocab_path} does not match ground-truth sha256/size")
        else:
            warnings.append(f"{resource_id}: {vocab_path} absent locally, file check skipped")

    for warning in warnings:
        print(f"[warn] {warning}")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        sys.exit(f"registry invalid: {len(errors)} error(s), {len(warnings)} warning(s)")

    n_langs = len({r["language"] for r in resources.values()})
    print(f"registry OK: {len(resources)} resources, {n_langs} languages, "
          f"{len(warnings)} warning(s) ({registry_path})")


if __name__ == "__main__":
    main()
