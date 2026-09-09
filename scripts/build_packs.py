#!/usr/bin/env python3
"""Build language packs: one fetch for a whole language (plan 113).

A language load today is five transfers from two repos (aff, dic from
kotoshu/dictionaries; mini model, vocab, buckets from this repo). A pack
concatenates those five artifacts into ONE file the media host serves
CORS-fetchable (the only host that serves the tier bytes to browsers),
so a cold load pays one round trip and one progress stream. The PR #28
owner sketch shaped the registry entry; the framing here is the
length-prefixed section stream the sketch refined into.

Format (KPK1) — pinned, versioned by the magic:

    offset 0   magic  b"KPK1"              4 bytes
    offset 4   u32 LE section_count
    then one framing per section, in this order:

        u32 LE payload_length               4 bytes
        u8     tag                          1 byte
               payload                      payload_length bytes
               sha256(payload)              32 bytes (per-section footer)

    tags: 1 aff, 2 dic, 3 model (mini tier .onnx), 4 vocab (.vocab.json),
          5 buckets (.onnx). Every section is REQUIRED except buckets.
    Offsets are computable by walking the framing; the registry entry
    records them anyway so a consumer can slice without a scan.

The pack payload is the EXACT bytes of its parts — no compression, no
transformation — so the pack verifies by checksum alone: each section
footer is the sha256 of its payload, and the model/vocab/buckets
footers must equal the sha256 values models/<lang>/tiers.json already
carries (the builder refuses to cut a pack from drifted artifacts).
The dictionary sections come from the pinned kotoshu/dictionaries
commit recorded as `dictionary_pin` — packs bridge the two repos, so
the pack pins both (plan 113).

Section sources must be UTF-8 text for aff/dic (the wasm load path
hands dictionary sources to the engine as strings, exactly like a
per-artifact fetch does today).

Writes packs/<lang>-<version>.bin and the pack descriptors in
packs/packs.json that scripts/build_registry.py merges into registry.json
as additive `kotoshu://packs/<lang>` entries (type "pack"). The registry
stays release-tag v1.5.0 on the branch — the owner cuts the release and
bumps the registry data revision when the packs are validated (plan 113).

Usage:
    python scripts/build_packs.py --lang en de pt

With --golden, build a pack from explicit section files and write the
bytes to stdout (no registry/descriptor side effects) — the
cross-implementation fixture tests pin the output sha256 from both this
builder and the kotoshu-rs pack reader against the same inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MAGIC = b"KPK1"

TAG_AFF = 1
TAG_DIC = 2
TAG_MODEL = 3
TAG_VOCAB = 4
TAG_BUCKETS = 5

# The five sections, in pack order.
SECTION_TAGS = (
    ("aff", TAG_AFF),
    ("dic", TAG_DIC),
    ("model", TAG_MODEL),
    ("vocab", TAG_VOCAB),
    ("buckets", TAG_BUCKETS),
)
REQUIRED = ("aff", "dic", "model", "vocab")

# The dictionary commit the worker resolves today
# (kotoshu-worker DEFAULT_DICT_PIN) — the pin a pack must reproduce.
DEFAULT_DICT_PIN = "1829a3e2e67dc7ffb38f8dcd2d3d2294b6a8580d"

PACKS_SPEC = "kotoshu.packs/v1"
MIN_ENGINE_VERSION = "0.5"  # the @kotoshu/wasm line that ships loadPack
# The registry release the pack versions with (its parts are the v1.5.0
# tier artifacts); the owner bumps this with the release when the packs
# are validated (plan 113: v1.6.0).
DEFAULT_PACK_VERSION = "1.5.0"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frame_section(tag: int, payload: bytes) -> bytes:
    """One section framing: u32 LE length + tag byte + payload + sha256."""
    return struct.pack("<IB", len(payload), tag) + payload + hashlib.sha256(payload).digest()


def build_pack(sections: dict[str, bytes | None]) -> bytes:
    """Assemble the pack from its five sections (order fixed, buckets
    optional). None skips the section; anything else must be bytes."""
    framed = []
    for name, tag in SECTION_TAGS:
        payload = sections[name]
        if payload is None:
            continue
        if name in REQUIRED and not payload:
            raise ValueError(f"section {name!r} is empty")
        framed.append(frame_section(tag, payload))
    # The count is the sections actually written (buckets may be absent).
    return MAGIC + struct.pack("<I", len(framed)) + b"".join(framed)


def parse_pack(data: bytes) -> dict[int, bytes]:
    """Walk the framing; verify magic, count and every section footer.
    Returns tag -> payload. Raises ValueError on any structural drift."""
    if len(data) < 8:
        raise ValueError(f"pack too short for a header ({len(data)} bytes)")
    if data[:4] != MAGIC:
        raise ValueError(f"bad magic {data[:4]!r} (expected {MAGIC!r})")
    (count,) = struct.unpack_from("<I", data, 4)
    pos = 8
    out: dict[int, bytes] = {}
    for _ in range(count):
        if pos + 5 > len(data):
            raise ValueError(f"truncated section header at offset {pos}")
        (length,) = struct.unpack_from("<I", data, pos)
        tag = data[pos + 4]
        pos += 5
        if pos + length + 32 > len(data):
            raise ValueError(f"truncated section {tag} at offset {pos}")
        payload = data[pos : pos + length]
        pos += length
        footer = data[pos : pos + 32]
        pos += 32
        if hashlib.sha256(payload).digest() != footer:
            raise ValueError(f"section {tag} sha256 footer mismatch")
        if tag in out:
            raise ValueError(f"duplicate section tag {tag}")
        out[tag] = payload
    if pos != len(data):
        raise ValueError(f"{len(data) - pos} trailing bytes after {count} sections")
    return out


def section_offsets(data: bytes) -> dict[int, tuple[int, int]]:
    """tag -> (payload offset, length), walking the same framing."""
    (count,) = struct.unpack_from("<I", data, 4)
    pos = 8
    out = {}
    for _ in range(count):
        (length,) = struct.unpack_from("<I", data, pos)
        tag = data[pos + 4]
        pos += 5
        out[tag] = (pos, length)
        pos += length + 32
    return out


def read_dict_source(dict_repo: Path, pin: str, lang: str, ext: str) -> bytes:
    """The exact bytes of {lang} index.{ext} at the pinned dictionaries
    commit — both on-disk layouts the worker resolves ({lang}/spelling/
    first, flat second), read through git so the checkout state cannot
    leak into the pack."""
    for rel in (f"{lang}/spelling/index.{ext}", f"{lang}/index.{ext}"):
        result = subprocess.run(
            ["git", "-C", str(dict_repo), "cat-file", "blob", f"{pin}:{rel}"],
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout
    sys.exit(f"error: no index.{ext} for {lang!r} at dictionaries pin {pin}")


def load_tiers(repo: Path, lang: str) -> dict:
    path = repo / "models" / lang / "tiers.json"
    try:
        tiers = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read {path}: {exc}")
    if tiers.get("language") != lang:
        sys.exit(f"error: {path} declares language {tiers.get('language')!r}")
    return tiers["tiers"]


def verify_tier(path: Path, sha: str, size: int, label: str) -> bytes:
    """Read a tier artifact and refuse anything that drifted from
    models/<lang>/tiers.json — a pack is only as good as its parts."""
    data = path.read_bytes()
    if len(data) != size:
        sys.exit(f"error: {label}: {path} is {len(data)} bytes, tiers.json says {size}")
    if sha256_bytes(data) != sha:
        sys.exit(f"error: {label}: {path} sha256 does not match models tiers.json")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", nargs="+", default=None, help="language codes (e.g. en de pt)")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--dict-repo", type=Path, default=None,
                    help="kotoshu/dictionaries checkout (default: sibling directory)")
    ap.add_argument("--dict-pin", default=DEFAULT_DICT_PIN,
                    help="dictionaries commit the pack is cut from")
    ap.add_argument("--tier", default="mini",
                    help="embedding tier packed alongside the dictionary (default mini)")
    ap.add_argument("--version", default=None,
                    help="pack version (default: derived per language from its tier version)")
    ap.add_argument("--skip-buckets", action="store_true",
                    help="cut the pack without the bucket sibling (debug only)")
    ap.add_argument("--golden", nargs=5, metavar=("AFF", "DIC", "MODEL", "VOCAB", "BUCKETS"),
                    help="build a one-off pack from explicit section FILES and write it to "
                         "stdout; no descriptor or registry side effects (fixture tests)")
    args = ap.parse_args()

    if args.golden:
        sections = {
            "aff": Path(args.golden[0]).read_bytes(),
            "dic": Path(args.golden[1]).read_bytes(),
            "model": Path(args.golden[2]).read_bytes(),
            "vocab": Path(args.golden[3]).read_bytes(),
            "buckets": Path(args.golden[4]).read_bytes(),
        }
        sys.stdout.buffer.write(build_pack(sections))
        return 0

    if not args.lang:
        ap.error("--lang is required (unless --golden is given)")

    repo = args.repo_root.resolve()
    dict_repo = args.dict_repo or (repo.parent / "dictionaries")
    if not dict_repo.is_dir():
        sys.exit(f"error: dictionaries checkout not found at {dict_repo} (use --dict-repo)")

    packs_dir = repo / "packs"
    packs_dir.mkdir(exist_ok=True)
    descriptor_path = packs_dir / "packs.json"
    descriptor = {
        "spec": PACKS_SPEC,
        "generated_at": iso_now(),
        "packs": {},
    }
    if descriptor_path.exists():
        old = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if old.get("spec") != PACKS_SPEC:
            sys.exit(f"error: {descriptor_path} declares spec {old.get('spec')!r}")
        descriptor["packs"] = old.get("packs", {})

    for lang in args.lang:
        stem = f"fasttext.{lang}.{args.tier}"
        tier_dir = repo / "models" / lang
        tiers = load_tiers(repo, lang)
        tier = tiers.get(args.tier)
        buckets = tiers.get("buckets")
        if tier is None:
            sys.exit(f"error: no {args.tier!r} tier in models/{lang}/tiers.json")
        if buckets is None and not args.skip_buckets:
            sys.exit(f"error: no buckets sibling in models/{lang}/tiers.json "
                     f"(--skip-buckets to cut without it)")

        aff = read_dict_source(dict_repo, args.dict_pin, lang, "aff")
        dic = read_dict_source(dict_repo, args.dict_pin, lang, "dic")
        for name, data in (("aff", aff), ("dic", dic)):
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                sys.exit(f"error: dictionary {name} for {lang!r} at pin {args.dict_pin} "
                         f"is not UTF-8 ({exc}); the pack text sections must be UTF-8")
        model = verify_tier(tier_dir / f"{stem}.onnx", tier["sha256"], tier["bytes"],
                            f"{lang} {args.tier} onnx")
        vocab = verify_tier(tier_dir / f"{stem}.vocab.json", tier["vocab_sha256"],
                            tier["vocab_bytes"], f"{lang} {args.tier} vocab")
        buckets_bytes = None
        if buckets is not None and not args.skip_buckets:
            buckets_bytes = verify_tier(tier_dir / f"fasttext.{lang}.buckets.onnx",
                                        buckets["sha256"], buckets["bytes"],
                                        f"{lang} buckets onnx")

        sections = {"aff": aff, "dic": dic, "model": model, "vocab": vocab,
                    "buckets": buckets_bytes}
        pack_bytes = build_pack(sections)

        # Round-trip before writing anything: the pack must parse, and
        # every section must be byte-identical to its source artifact.
        parsed = parse_pack(pack_bytes)
        offsets = section_offsets(pack_bytes)
        tag_of = dict((tag, name) for name, tag in SECTION_TAGS)
        for name, tag in SECTION_TAGS:
            if sections[name] is None:
                if tag in parsed:
                    sys.exit(f"internal error: {lang} section {name} present but skipped")
                continue
            if parsed.get(tag) != sections[name]:
                sys.exit(f"internal error: {lang} section {name} round-trip mismatch")

        # The pack versions with the registry release its parts come from
        # (the registry build resolves the same value from release_tag,
        # so the descriptor and registry cannot disagree).
        version = args.version or DEFAULT_PACK_VERSION

        contents = {}
        for name, tag in SECTION_TAGS:
            payload = sections[name]
            if payload is None:
                continue
            offset, length = offsets[tag]
            contents[name] = {
                "tag": tag,
                "offset": offset,
                "length": length,
                "sha256": sha256_bytes(payload),
            }

        descriptor["packs"][lang] = {
            "language": lang,
            "version": version,
            "tier": args.tier,
            "dictionary_pin": args.dict_pin,
            "contents": contents,
            "sha256": sha256_bytes(pack_bytes),
            "size_bytes": len(pack_bytes),
            "licenses": {
                "embeddings": "CC-BY-SA-3.0",
                "dictionary": (f"https://cdn.jsdelivr.net/gh/kotoshu/dictionaries@"
                               f"{args.dict_pin}/{lang}/spelling/license"),
            },
            "min_engine_version": MIN_ENGINE_VERSION,
        }

        out_path = packs_dir / f"{lang}-{version}.bin"
        out_path.write_bytes(pack_bytes)
        print(f"[{lang}] wrote {out_path.relative_to(repo)} "
              f"({len(pack_bytes):,} bytes, sections: "
              f"{', '.join(n for n, _ in SECTION_TAGS if sections[n] is not None)})")

    descriptor_path.write_text(json.dumps(descriptor, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    print(f"wrote {descriptor_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
