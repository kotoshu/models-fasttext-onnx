#!/usr/bin/env python3
"""Validate registry.json.

Always checks schema conformance (schemas/registry.schema.json), id
uniqueness, URL construction, and consistency with the ground-truth
sources (manifest.json for the full tier, models/<lang>/tiers.json for
derived tiers, packs/packs.json + tiers.json for language packs).
--check-files additionally hashes local model and vocab files when
present; absent files are warnings (e.g. CI before a tier build),
mismatches are failures. Exits nonzero on any failure.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

# Pack framing shared with scripts/build_packs.py (plan 113).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_packs import (  # noqa: E402
    TAG_AFF, TAG_BUCKETS, TAG_DIC, TAG_MODEL, TAG_VOCAB, parse_pack, section_offsets,
)

REPO_URL = "https://github.com/kotoshu/models-fasttext-onnx"
MEDIA_URL = "https://media.githubusercontent.com/media/kotoshu/models-fasttext-onnx"
READ_CHUNK = 1 << 20

PACKS_DESCRIPTOR = "packs/packs.json"
PACK_SECTION_TIER = {  # contents name -> (tiers.json tier, field pair)
    "model": ("tier", "sha256", "bytes"),
    "vocab": ("tier", "vocab_sha256", "vocab_bytes"),
    "buckets": ("buckets", "sha256", "bytes"),
}

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


CONTENT_RANGE_TOTAL = re.compile(r"bytes\s+\d+-\d+/(\d+)")
URL_TIMEOUT_S = 30
URL_PROBE_WORKERS = 8


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def probe_url(url):
    """Ranged GET (1 byte) so the media host's HEAD quirks never matter.

    Returns ("ok", total_bytes) or (reason, None). total_bytes comes from
    Content-Range (206) or Content-Length (200) and is None when unknown.
    4xx is immediately fatal; 5xx/network errors retry with backoff
    (GitHub release downloads occasionally 500 under load).
    """
    last_reason = "unknown error"
    for attempt in range(3):
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        try:
            with urllib.request.urlopen(request, timeout=URL_TIMEOUT_S) as response:
                total = None
                content_range = response.headers.get("Content-Range")
                if content_range:
                    match = CONTENT_RANGE_TOTAL.match(content_range)
                    total = int(match.group(1)) if match else None
                elif response.status == 200:
                    length = response.headers.get("Content-Length")
                    total = int(length) if length else None
                return "ok", total
        except urllib.error.HTTPError as exc:
            last_reason = f"HTTP {exc.code}"
            if exc.code < 500:
                break
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_reason = str(exc) or exc.__class__.__name__
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))
    return last_reason, None


def mirror_probe_url(url, ref):
    """Probe a mirror URL at the validating ref, not the /main/ it names.

    LFS objects are content-addressed and exist at every ref carrying
    them, while an artifact added on a PR branch 404s on main until the
    merge — so branch validation must probe the branch's copy.
    """
    if ref and ref != "main":
        return url.replace(f"{MEDIA_URL}/main/", f"{MEDIA_URL}/{ref}/")
    return url


def check_urls_live(resources, errors, ref="main"):
    """Probe every non-null primary/mirror/vocab URL (plan 10).

    The mirror convention assumes LFS-tracked artifacts: the media host
    only serves LFS objects, so a plain-git blob yields a silent 404 that
    construction-only URL checks cannot see. Vocab sidecars are probed for
    reachability only — their authoritative sizes live in ground truth.
    """
    probes = []
    for resource_id, resource in resources.items():
        urls = resource.get("urls") or {}
        for kind in ("primary", "mirror"):
            if urls.get(kind):
                probes.append((resource_id, kind, mirror_probe_url(urls[kind], ref),
                               resource.get("size_bytes")))
        if resource.get("vocab_url"):
            probes.append((resource_id, "vocab", mirror_probe_url(resource["vocab_url"], ref), None))

    with ThreadPoolExecutor(max_workers=URL_PROBE_WORKERS) as pool:
        results = list(pool.map(lambda p: (p, probe_url(p[2])), probes))

    for (resource_id, kind, url, expected_size), (status, total) in sorted(results):
        if status != "ok":
            errors.append(f"{resource_id}: {kind} URL unreachable ({status}): {url}")
        elif expected_size is not None and total is not None and total != expected_size:
            errors.append(
                f"{resource_id}: {kind} URL serves {total} bytes, registry declares {expected_size}"
            )
    return len(probes)


def file_sha256_and_size(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def asset_stems(lang, tier_name):
    if tier_name == "lid-176":  # plan 102: the LID pair mirrors upstream lid.176 naming
        return "lid.176.onnx", "lid.176.vocab.json"
    if tier_name == "typo-biencoder":  # plan 115: the typo pair lives under models/typo/
        return "typo.biencoder.onnx", "typo.biencoder.vocab.json"
    stem = f"fasttext.{lang}" if tier_name == "full" else f"fasttext.{lang}.{tier_name}"
    # Buckets (plan 103) have no vocab sibling - the bucket_ids tensor is inside.
    if tier_name == "buckets":
        return f"{stem}.onnx", None
    return f"{stem}.onnx", f"{stem}.vocab.json"


def check_urls(resource, resource_id, registry, errors):
    lang = resource["language"]
    tag = registry["release_tag"]

    if resource["type"] == "pack":
        # Plan 113: one LFS-committed section stream under packs/. The
        # mirror is the browser-usable artifact; packs ship as release
        # assets only when the owner cuts the pack release (plan 113),
        # so primary must stay null until then.
        name = f"{lang}-{resource['version']}.bin"
        expected_mirror = f"{MEDIA_URL}/main/packs/{name}"
        if resource["urls"]["mirror"] != expected_mirror:
            errors.append(f"{resource_id}: mirror URL expected {expected_mirror}")
        if resource["urls"]["primary"] is not None:
            errors.append(f"{resource_id}: pack primary URL set but no pack release "
                          f"exists (plan 113 keeps packs media-host only)")
        return

    tier_name = resource["tier"]["name"]
    onnx_name, vocab_name = asset_stems(lang, tier_name)

    if tier_name == "typo-matrix":
        # Plan 136: the prebuilt KTM1 retrieval matrix. Mirror-only
        # until a release carries it (the additive template); no vocab
        # sibling (rows pair with the language's full-tier vocab).
        expected = f"{MEDIA_URL}/main/models/{lang}/typo.matrix.{lang}.ktm1"
        if resource["urls"]["mirror"] != expected:
            errors.append(f"{resource_id}: mirror URL expected {expected}")
        if resource["urls"]["primary"] is not None or resource["vocab_url"] is not None:
            errors.append(f"{resource_id}: typo-matrix rides mirror-only until a "
                          f"release carries the artifact")
        # Plan 14: the rows are index-parallel to EXACTLY the tier vocab
        # they were derived over; a rebuilt full tier silently
        # mismatches every row. The entry must carry the pairing sha,
        # and check_urls' ground truth compares it to the live tier.
        import re as _re
        paired = resource.get("paired_vocab_sha256")
        if not (isinstance(paired, str) and _re.fullmatch(r"[0-9a-f]{64}", paired)):
            errors.append(f"{resource_id}: paired_vocab_sha256 missing or malformed")
        return

    if tier_name == "typo-biencoder":
        # Plan 115/131: the typo bi-encoder under models/typo/ rides the
        # descriptor's release_tag (the owner's knob). Unset: mirror-only,
        # primary/vocab null (the plan-113 additive template). Set: BOTH
        # URLs must follow the release-tag convention exactly — a primary
        # without its vocab sibling is a dead half-pair.
        expected_mirror = f"{MEDIA_URL}/main/models/typo/{onnx_name}"
        if resource["urls"]["mirror"] != expected_mirror:
            errors.append(f"{resource_id}: mirror URL expected {expected_mirror}")
        primary = resource["urls"]["primary"]
        vocab_url = resource["vocab_url"]
        if primary is None and vocab_url is None:
            return
        # Plan 136: the pair pins to ITS OWN release tag (the cut that
        # carried the assets), which can be older than this registry's
        # release_tag - only the pairing must hold: both URLs at the
        # SAME semver tag under the release download path.
        import re as _re
        pair = _re.fullmatch(
            r"https://github\.com/kotoshu/models-fasttext-onnx/releases/download/"
            r"(v\d+\.\d+\.\d+)/typo\.biencoder\.onnx", primary or "")
        if (pair and vocab_url ==
                f"https://github.com/kotoshu/models-fasttext-onnx/releases/download/"
                f"{pair.group(1)}/typo.biencoder.vocab.json"):
            return
        errors.append(f"{resource_id}: typo primary/vocab must either both be null "
                      f"(pre-release, mirror serves) or both sit at the SAME semver "
                      f"release tag; got primary={primary!r} vocab={vocab_url!r}")
        return

    # Every tier binary is an LFS object in git -> the media host mirror
    # (the raw host serves pointer stubs). All tiers follow one rule.
    # LID lives under models/lid/, not models/{lang}/.
    if tier_name == "lid-176":
        expected_mirror = f"{MEDIA_URL}/main/models/lid/{onnx_name}"
    else:
        expected_mirror = f"{MEDIA_URL}/main/models/{lang}/{onnx_name}"
    if resource["urls"]["mirror"] != expected_mirror:
        errors.append(f"{resource_id}: mirror URL expected {expected_mirror}")

    if tag is None:
        if resource["urls"]["primary"] is not None or resource["vocab_url"] is not None:
            errors.append(f"{resource_id}: primary/vocab URLs set but release_tag is null")
    else:
        expected_primary = f"{REPO_URL}/releases/download/{tag}/{onnx_name}"
        if resource["urls"]["primary"] != expected_primary:
            errors.append(f"{resource_id}: primary URL expected {expected_primary}")
        # Buckets carry no vocab sibling (plan 103).
        if vocab_name is None:
            if resource.get("vocab_url") is not None:
                errors.append(f"{resource_id}: vocab_url must be null for buckets")
        else:
            expected_vocab = f"{REPO_URL}/releases/download/{tag}/{vocab_name}"
            if resource["vocab_url"] != expected_vocab:
                errors.append(f"{resource_id}: vocab_url expected {expected_vocab}")


def check_ground_truth(resource, resource_id, root, manifest, errors):
    lang = resource["language"]
    tier_name = resource["tier"]["name"]

    if resource["type"] == "pack":
        return check_pack_ground_truth(resource, resource_id, root, errors)

    if tier_name == "buckets":
        # Ground truth is models/<lang>/tiers.json buckets entry; no vocab.
        tiers_path = root / "models" / lang / "tiers.json"
        try:
            tiers = load_json(tiers_path)
            t = tiers["tiers"]["buckets"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{resource_id}: cannot read buckets ground truth: {exc}")
            return None
        if resource["sha256"] != t["sha256"] or resource["size_bytes"] != t["bytes"]:
            errors.append(f"{resource_id}: sha256/size drift vs models/{lang}/tiers.json")
        return None

    if tier_name == "typo-biencoder":  # plan 115: models/typo/typo.json is the ground truth
        try:
            d = load_json(root / "models" / "typo" / "typo.json")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{resource_id}: cannot read models/typo/typo.json: {exc}")
            return None
        if resource["sha256"] != d["sha256"] or resource["size_bytes"] != d["bytes"]:
            errors.append(f"{resource_id}: sha256/size drift vs models/typo/typo.json")
        return (d["vocab_sha256"], d["vocab_bytes"])

    if tier_name == "lid-176":  # plan 102: models/lid/lid.json is the ground truth
        try:
            d = load_json(root / "models" / "lid" / "lid.json")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{resource_id}: cannot read models/lid/lid.json: {exc}")
            return None
        if resource["sha256"] != d["sha256"] or resource["size_bytes"] != d["bytes"]:
            errors.append(f"{resource_id}: sha256/size drift vs models/lid/lid.json")
        return (d["vocab_sha256"], d["vocab_bytes"])

    if tier_name == "full":
        entry = manifest["resources"].get(f"models/{lang}/fasttext.{lang}.onnx")
        if entry is None:
            errors.append(f"{resource_id}: no manifest.json entry for models/{lang}/fasttext.{lang}.onnx")
            return None
        if resource["sha256"] != entry["sha256"] or resource["size_bytes"] != entry["size"]:
            errors.append(f"{resource_id}: sha256/size drift vs manifest.json")
        vocab_entry = manifest["resources"].get(f"models/{lang}/fasttext.{lang}.vocab.json")
        return None if vocab_entry is None else (vocab_entry["sha256"], vocab_entry["size"])

    # Plan 136: the prebuilt matrix is descriptor-driven
    # (models/{lang}/typo-matrix.json), not a tiers.json tier.
    if tier_name == "typo-matrix":
        desc = load_json(root / "models" / lang / "typo-matrix.json")
        if resource["sha256"] != desc["sha256"] or resource["size_bytes"] != desc["bytes"]:
            errors.append(f"{resource_id}: sha256/size drift vs models/{lang}/typo-matrix.json")
        # Plan 14: cross-check the pairing against the full tier's
        # ground truth (the manifest) - a rebuilt tier without a
        # rebuilt matrix ships silently-wrong rows.
        if manifest is not None:
            full = (manifest.get("resources") or {}).get(f"models/{lang}/fasttext.{lang}.onnx")
            if full and resource.get("paired_vocab_sha256") != full["sha256"]:
                errors.append(
                    f"{resource_id}: pairs with tier {str(resource.get('paired_vocab_sha256'))[:12]} "
                    f"but the full tier on disk is {full['sha256'][:12]} - rebuild the matrix "
                    f"(scripts/build_typo_matrices.py)")
        return None

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


def check_pack_ground_truth(resource, resource_id, root, errors):
    """Plan 113 pack entries: the descriptor (packs/packs.json) is the
    ground truth for the whole entry, and models/<lang>/tiers.json is
    the ground truth for the model/vocab/buckets section checksums -
    the pack must embed exactly the artifacts the registry serves."""
    descriptor_path = root / PACKS_DESCRIPTOR
    try:
        descriptor = load_json(descriptor_path)
        pack = descriptor["packs"][resource["language"]]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"{resource_id}: cannot read pack descriptor {descriptor_path}: {exc}")
        return
    for field in ("version", "tier", "dictionary_pin", "sha256", "size_bytes",
                  "contents", "min_engine_version"):
        if resource.get(field) != pack.get(field):
            errors.append(f"{resource_id}: {field} drift vs {PACKS_DESCRIPTOR}")

    tiers_path = root / "models" / resource["language"] / "tiers.json"
    try:
        tiers = load_json(tiers_path)["tiers"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"{resource_id}: cannot read {tiers_path}: {exc}")
        return
    contents = resource["contents"]
    for name, (tier_key, sha_field, size_field) in PACK_SECTION_TIER.items():
        if name not in contents:
            continue  # buckets is optional in the pack
        try:
            truth = tiers[tier_key] if tier_key == "buckets" else tiers[resource["tier"]]
        except KeyError:
            errors.append(f"{resource_id}: no {tier_key!r} tier in {tiers_path} "
                          f"for pack section {name!r}")
            continue
        section = contents[name]
        if section["sha256"] != truth[sha_field] or section["length"] != truth[size_field]:
            errors.append(f"{resource_id}: section {name!r} sha256/size drift vs "
                          f"{tiers_path} {tier_key!r}")


def check_pack_file(resource, resource_id, root, errors, warnings):
    """--check-files for packs: hash the local pack file AND walk its
    framing - magic, count, per-section sha256 footers, and the
    offsets/lengths/sha256 the registry entry declares. Absent file is
    a warning, like the tier files."""
    name = f"{resource['language']}-{resource['version']}.bin"
    path = root / "packs" / name
    if not path.exists():
        warnings.append(f"{resource_id}: {path} absent locally, file check skipped")
        return
    data = path.read_bytes()
    sha, size = hashlib.sha256(data).hexdigest(), len(data)
    if sha != resource["sha256"] or size != resource["size_bytes"]:
        errors.append(f"{resource_id}: local pack {path} does not match registry sha256/size")
        return
    try:
        parsed = parse_pack(data)  # verifies every section footer
        offsets = section_offsets(data)
    except ValueError as exc:
        errors.append(f"{resource_id}: pack {path} framing invalid: {exc}")
        return
    tag_of = {TAG_AFF: "aff", TAG_DIC: "dic", TAG_MODEL: "model",
              TAG_VOCAB: "vocab", TAG_BUCKETS: "buckets"}
    seen = {tag_of[tag] for tag in parsed}
    for section_name, section in resource["contents"].items():
        if section_name not in seen:
            errors.append(f"{resource_id}: declared section {section_name!r} missing "
                          f"from {path}")
            continue
        offset, length = offsets[section["tag"]]
        if (offset, length) != (section["offset"], section["length"]):
            errors.append(f"{resource_id}: section {section_name!r} offset/length "
                          f"{(offset, length)} != declared "
                          f"{(section['offset'], section['length'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", help="path to registry.json (default: <repo-root>/registry.json)")
    ap.add_argument("--repo-root", default=".", help="repository root (default: current directory)")
    ap.add_argument("--schema", help="path to registry.schema.json (default: <repo-root>/schemas/registry.schema.json)")
    ap.add_argument("--check-files", action="store_true",
                    help="hash local model/vocab files present on disk (absent files are warnings)")
    ap.add_argument("--check-urls", action="store_true",
                    help="probe registry URLs with ranged GETs (network); "
                         "unreachable URLs and size mismatches are failures")
    ap.add_argument("--urls-ref", default="main",
                    help="git ref to probe mirror URLs against (pass the branch "
                         "on PRs: artifacts added there do not exist on main yet)")
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
        if resource["type"] == "pack":
            # The manifest has nothing to say about packs (their parts are
            # dictionaries-repo + tiers); the pack descriptor does.
            check_pack_ground_truth(resource, resource_id, root, errors)
            if args.check_files:
                check_pack_file(resource, resource_id, root, errors, warnings)
            continue
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

        if vocab_name is None:
            continue
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

    probed = 0
    if args.check_urls and not errors:
        probed = check_urls_live(resources, errors, ref=args.urls_ref)

    for warning in warnings:
        print(f"[warn] {warning}")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        sys.exit(f"registry invalid: {len(errors)} error(s), {len(warnings)} warning(s)")

    n_langs = len({r["language"] for r in resources.values()})
    print(f"registry OK: {len(resources)} resources, {n_langs} languages, "
          f"{len(warnings)} warning(s) ({registry_path})")
    if probed:
        print(f"URLs OK: {probed} URL(s) reachable")


if __name__ == "__main__":
    main()
