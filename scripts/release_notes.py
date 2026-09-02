#!/usr/bin/env python3
"""Print markdown release notes for a built registry.json to stdout.

Per-language tier tables come from the registry; header facts (opset,
upstream sources) come from manifest.json next to the registry.
"""

import argparse
import json
from pathlib import Path

# ja/ko/zh manifest entries predate opset recording; the conversion
# pipeline is uniform, so they fall back to this value.
DEFAULT_OPSET = 11
TIER_ORDER = {"mini": 0, "fluency": 1, "full": 2}
LICENSE_LINE = "CC-BY-SA-3.0 (derived from FastText pretrained vectors)"


def human_size(n):
    mib = n / (1 << 20)
    if mib >= 1024:
        return f"{mib / 1024:.1f} GiB"
    return f"{mib:.1f} MiB"


def manifest_facts(manifest):
    opsets = set()
    sources = set()
    for entry in manifest["resources"].values():
        if entry.get("type") != "onnx":
            continue
        opsets.add(entry.get("opset_version", DEFAULT_OPSET))
        sources.add(entry.get("source") or entry.get("fasttext_source") or "FastText Common Crawl")
    return sorted(opsets), sorted(sources)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default="registry.json", help="path to registry.json")
    args = ap.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    manifest = json.loads((registry_path.parent / "manifest.json").read_text(encoding="utf-8"))

    resources = registry["resources"]
    languages = sorted({r["language"] for r in resources.values()})
    total_bytes = sum(r["size_bytes"] for r in resources.values())
    opsets, sources = manifest_facts(manifest)
    tag = registry.get("release_tag")

    print(f"# Kotoshu ONNX models {tag or '(dev)'}")
    print()
    print(f"- Languages: {len(languages)} ({', '.join(languages)})")
    print(f"- Resources: {len(resources)} models")
    print(f"- Total size: {human_size(total_bytes)}")
    print(f"- ONNX opset: {', '.join(str(o) for o in opsets)}")
    print(f"- Upstream: {', '.join(sources)}")
    print(f"- License: {LICENSE_LINE}")
    print("- Tier quality is gated by measured evals; per-tier reports are linked below.")
    print()

    for lang in languages:
        print(f"## {lang}")
        print()
        print("| Tier | Asset | Size | Eval |")
        print("|---|---|---|---|")
        rows = sorted(
            (r for r in resources.values() if r["language"] == lang),
            key=lambda r: TIER_ORDER[r["tier"]["name"]],
        )
        for r in rows:
            asset = r["urls"]["mirror"].rsplit("/", 1)[-1]
            print(f"| {r['tier']['name']} | `{asset}` | {human_size(r['size_bytes'])} | {r['eval_ref'] or '—'} |")
        print()


if __name__ == "__main__":
    main()
