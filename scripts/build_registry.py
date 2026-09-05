#!/usr/bin/env python3
"""Build registry.json (Resource Spec v1) at the repo root.

Merges manifest.json (the full tier) with per-language
models/<lang>/tiers.json (mini/fluency tiers). tiers.json files are
produced by scripts/build_tiers.py; a missing file is skipped silently
unless --strict, which is the release-time mode.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SPEC = "kotoshu.resources/v1"
# Registry data revision (the spec id above is the format version).
# 1: v1.0.x releases, 9 languages. 2: plan 77 coverage expansion, 13 new
# languages under release tag v1.1.0 (minor = coverage per plan 05).
# 3: plan 83 batch 2, gem-wired RTL languages plus national-script and
# Latin newcomers under release tag v1.2.0.
REGISTRY_VERSION = 4
REPO_URL = "https://github.com/kotoshu/models-fasttext-onnx"
# LFS-tracked binaries resolve to pointer stubs on the raw host; the
# media host serves the real bytes. Plain-git files (vocab, manifests)
# are fine on raw.
MEDIA_URL = "https://media.githubusercontent.com/media/kotoshu/models-fasttext-onnx"
LICENSE = "CC-BY-SA-3.0"
MIN_ENGINE_VERSION = "0.7"
FULL_DIMS = 300
KNOWN_TIERS = ("fluency", "mini")
DEV_VERSION = "0.0.0-dev"


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def full_vocab_size(lang_dir, lang):
    path = lang_dir / f"fasttext.{lang}.vocab.json"
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read {path}: {exc}")
    if isinstance(data, dict) and "vocab_size" in data:
        return data["vocab_size"]
    return len(data)


def build_resource(lang, tier_name, dims, vocab_size, quantization,
                   sha256, size_bytes, eval_ref, version, tag):
    stem = f"fasttext.{lang}" if tier_name == "full" else f"fasttext.{lang}.{tier_name}"
    # Every tier binary lives in git as an LFS object (plan 92) so the
    # media host serves CORS-fetchable bytes for browsers; the raw host
    # would serve 134-byte pointer stubs, hence the media URL.
    mirror = f"{MEDIA_URL}/main/models/{lang}/{stem}.onnx"
    return {
        "type": "model",
        "language": lang,
        "tier": {
            "name": tier_name,
            "dims": dims,
            "vocab_size": vocab_size,
            "quantization": quantization,
        },
        "version": version,
        "urls": {
            "primary": f"{REPO_URL}/releases/download/{tag}/{stem}.onnx" if tag else None,
            "mirror": mirror,
        },
        "vocab_url": f"{REPO_URL}/releases/download/{tag}/{stem}.vocab.json" if tag else None,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "license": LICENSE,
        "min_engine_version": MIN_ENGINE_VERSION,
        "eval_ref": eval_ref,
    }


def load_tiers(root, lang, strict):
    path = root / "models" / lang / "tiers.json"
    if not path.exists():
        if strict:
            sys.exit(f"error: {path} missing in strict mode; run scripts/build_tiers.py --lang {lang} first")
        return None
    try:
        tiers = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read {path}: {exc}")
    if tiers.get("language") != lang:
        sys.exit(f"error: {path} declares language {tiers.get('language')!r}, expected {lang!r}")
    return tiers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".", help="repository root (default: current directory)")
    ap.add_argument("--tag", help="release tag; without it a dev registry (no release URLs) is produced")
    ap.add_argument("--strict", action="store_true",
                    help="treat missing tiers.json as an error instead of skipping the language")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read {manifest_path}: {exc}")

    # Registry versions drop the leading "v" of the tag to match the dev
    # flavor ("0.0.0-dev") and the plan's v1.0.0 -> 1.0.0 example.
    version = (args.tag[1:] if args.tag.startswith("v") else args.tag) if args.tag else DEV_VERSION

    languages = sorted({
        path.split("/")[1]
        for path, entry in manifest["resources"].items()
        if entry.get("type") == "onnx"
    })

    resources = {}
    for lang in languages:
        try:
            entry = manifest["resources"][f"models/{lang}/fasttext.{lang}.onnx"]
        except KeyError:
            sys.exit(f"error: manifest.json has vocab but no onnx entry for language {lang!r}")
        resources[f"kotoshu://models/{lang}/full"] = build_resource(
            lang, "full", FULL_DIMS, full_vocab_size(root / "models" / lang, lang),
            None, entry["sha256"], entry["size"], None, version, args.tag)

        tiers = load_tiers(root, lang, args.strict)
        if tiers is None:
            continue
        for tier_name in sorted(tiers.get("tiers", {})):
            if tier_name not in KNOWN_TIERS:
                message = f"{root / 'models' / lang / 'tiers.json'}: unknown tier {tier_name!r}"
                if args.strict:
                    sys.exit(f"error: {message}")
                print(f"warning: {message} skipped", file=sys.stderr)
                continue
            t = tiers["tiers"][tier_name]
            try:
                resources[f"kotoshu://models/{lang}/{tier_name}"] = build_resource(
                    lang, tier_name, t["dims"], t["vocab_size"], t["quantization"],
                    t["sha256"], t["bytes"], t.get("eval_ref"), version, args.tag)
            except KeyError as exc:
                sys.exit(f"error: models/{lang}/tiers.json tier {tier_name!r} missing field {exc}")

    registry = {
        "spec": SPEC,
        "registry_version": REGISTRY_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "release_tag": args.tag,
        "resources": dict(sorted(resources.items())),
    }
    out = root / "registry.json"
    out.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tier_count = len({r["tier"]["name"] for r in registry["resources"].values()})
    print(f"Wrote {out}: {len(registry['resources'])} resources, "
          f"{len(languages)} languages, {tier_count} tier kinds, tag={args.tag or 'dev'}")


if __name__ == "__main__":
    main()
