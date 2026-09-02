#!/usr/bin/env python3
"""Build nested (2D-Matryoshka) single-artifact models — plan 71.

One file per language: the FULL 100k x 300d vocab, int8-per-row
quantized with the exact same math as the shipped tiers (functions are
imported from build_tiers.py, never duplicated). Because the
quantization is per-row, the first-N row-prefix view of the nested
artifact is bit-identical to the separately shipped tier built from the
same matrix rows:

    fasttext.{lang}.nested.onnx[:mini]    == fasttext.{lang}.mini.onnx
    fasttext.{lang}.nested.onnx[:fluency] == fasttext.{lang}.fluency.onnx

Per language this writes:

- models/{lang}/fasttext.{lang}.nested.onnx — full-vocab int8 artifact
  (~30 MB), metadata_props carry the truncation manifest (tier ->
  prefix length, read from models/{lang}/tiers.json).
- models/{lang}/fasttext.{lang}.nested.json — small committable
  manifest sibling {"truncations": {...}, "sha256": ..., "bytes": ...}.
- eval/reports/nested.summary.json — size report across languages
  (separate tier bytes vs nested vs full) plus the recommendation.

It also asserts, at build time, that each tier view is array-equal to
the shipped tier file's q_embeddings/row_scale (parity is re-verified
end-to-end, including metric parity, by check_nested_parity.py).

Registry/manifest/release are untouched — this is an experiment with a
recommendation, not an adoption (see TODO.impl plan 71).
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import StringStringEntryProto

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tiers  # noqa: E402


def nested_truncations(repo: Path, lang: str) -> dict[str, int]:
    """Prefix lengths per tier, straight from the shipped tiers.json."""
    tiers = json.loads((repo / "models" / lang / "tiers.json").read_text(encoding="utf-8"))
    return {tier: int(entry["vocab_size"]) for tier, entry in sorted(tiers["tiers"].items())}


def verify_nested_onnx(onnx_path: Path, dequant: np.ndarray, truncations: dict[str, int]) -> dict:
    """Runtime spot check the full-vocab artifact (incl. each tier cutoff)."""
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    idxs = sorted({0, 1, dequant.shape[0] // 2, dequant.shape[0] - 1} | {n - 1 for n in truncations.values()})
    worst_runtime = 0.0
    for i in idxs:
        out = session.run(["embedding"], {"word_index": np.array([i], dtype=np.int64)})[0]
        worst_runtime = max(worst_runtime, float(np.abs(out - dequant[i]).max()))
    if worst_runtime > 1e-4:
        raise RuntimeError(f"{onnx_path}: onnxruntime diverges from dequantized rows (max abs {worst_runtime:.3g})")
    return {"runtime_vs_dequant_max_abs": worst_runtime, "spot_checked_indices": idxs}


def assert_view_parity(
    nested_q: np.ndarray,
    nested_scale: np.ndarray,
    shipped_onnx: Path,
    tier: str,
    n: int,
) -> None:
    """The tier view must be array-equal to the shipped tier artifact."""
    from onnx import numpy_helper

    shipped = onnx.load(str(shipped_onnx))
    sq = ss = None
    for init in shipped.graph.initializer:
        if init.name == "q_embeddings":
            sq = numpy_helper.to_array(init)
        elif init.name == "row_scale":
            ss = numpy_helper.to_array(init)
    if sq is None or ss is None:
        for node in shipped.graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name != "value":
                    continue
                if attr.t.name == "q_embeddings" or "q_embeddings" in node.output:
                    sq = numpy_helper.to_array(attr.t)
                elif attr.t.name == "row_scale" or "row_scale" in node.output:
                    ss = numpy_helper.to_array(attr.t)
    if sq is None or ss is None:
        raise KeyError(f"{shipped_onnx}: q_embeddings/row_scale not found")
    if sq.shape[0] != n:
        raise ValueError(f"{shipped_onnx}: shipped vocab {sq.shape[0]} != truncation {n}")
    if not np.array_equal(nested_q[:n], sq):
        raise AssertionError(f"{tier}: nested q_embeddings[:{n}] != shipped {shipped_onnx.name}")
    if not np.array_equal(nested_scale[:n], ss):
        raise AssertionError(f"{tier}: nested row_scale[:{n}] != shipped {shipped_onnx.name}")


def build_language(repo: Path, lang: str) -> dict:
    model_dir = repo / "models" / lang
    print(f"[{lang}] loading full model")
    full = onnx.load(str(model_dir / f"fasttext.{lang}.onnx"))
    x = np.ascontiguousarray(build_tiers.constant_array(full, "word_embeddings"), dtype=np.float32)
    vocab = json.loads((model_dir / f"fasttext.{lang}.vocab.json").read_text(encoding="utf-8"))
    if x.shape[0] != vocab["vocab_size"] or len(vocab["word_to_idx"]) != x.shape[0]:
        raise ValueError(f"{lang}: vocab json does not match embedding matrix {x.shape}")
    source_sha = build_tiers.full_model_sha256(repo, lang)
    truncations = nested_truncations(repo, lang)

    # Same recipe as the shipped tiers: full dims, per-row int8.
    projected = build_tiers.reduce_dims_uncentered(x, 300)
    q, scale = build_tiers.quantize_int8_per_row(projected)
    dequant = q.astype(np.float32) * scale[:, None]
    quant_max_abs = float(np.abs(projected - dequant).max())

    model = build_tiers.make_tier_model(q, scale, "nested")
    model.metadata_props.append(
        StringStringEntryProto(key="truncations", value=json.dumps(truncations, separators=(",", ":")))
    )
    model.metadata_props.append(StringStringEntryProto(key="source_full_sha256", value=source_sha))

    onnx_path = model_dir / f"fasttext.{lang}.nested.onnx"
    onnx.save(model, str(onnx_path))

    check = verify_nested_onnx(onnx_path, dequant, truncations)
    check["dequant_vs_projected_max_abs"] = quant_max_abs  # informational for rows beyond the fluency cut

    for tier, n in truncations.items():
        assert_view_parity(q, scale, model_dir / f"fasttext.{lang}.{tier}.onnx", tier, n)
        print(f"  view parity vs shipped {tier} ({n} rows): OK")

    sibling = {
        "language": lang,
        "truncations": truncations,
        "vocab_size": int(q.shape[0]),
        "dims": int(q.shape[1]),
        "quantization": "int8-per-row",
        "bytes": onnx_path.stat().st_size,
        "sha256": sha256(onnx_path.read_bytes()).hexdigest(),
        "source_full_sha256": source_sha,
        "check": check,
        "generated_at": build_tiers.iso_now(),
    }
    (model_dir / f"fasttext.{lang}.nested.json").write_text(json.dumps(sibling, indent=2) + "\n", encoding="utf-8")

    tier_bytes = {
        tier: (model_dir / f"fasttext.{lang}.{tier}.onnx").stat().st_size for tier in truncations
    }
    return {
        "truncations": truncations,
        "mini_bytes": tier_bytes["mini"],
        "fluency_bytes": tier_bytes["fluency"],
        "separate_bytes": tier_bytes["mini"] + tier_bytes["fluency"],
        "nested_bytes": sibling["bytes"],
        "full_bytes": (model_dir / f"fasttext.{lang}.onnx").stat().st_size,
    }


def recommendation(sizes: dict[str, dict]) -> str:
    """One-liner from per-language figures (uniform across languages)."""
    mini = next(iter(sizes.values()))["mini_bytes"]
    flu_lo = min(v["fluency_bytes"] for v in sizes.values())
    flu_hi = max(v["fluency_bytes"] for v in sizes.values())
    nested = next(iter(sizes.values()))["nested_bytes"]
    full = next(iter(sizes.values()))["full_bytes"]
    return (
        "Keep the separate tiers as the default download: a mini-only user pays "
        f"~{mini / 1e6:.0f} MB today but would pay ~{nested / 1e6:.0f} MB "
        f"({nested / mini:.0f}x) for the nested file, and both tiers together "
        f"(~{flu_lo / 1e6:.0f}-{flu_hi / 1e6:.0f} + 3 MB) are still smaller than nested. "
        "Ship the nested artifact as the single optional all-in-one file instead: one download per language, "
        "mini->fluency upgrades are free (bytes already on disk), and it uniquely offers full-100k-vocab "
        f"coverage at ~{nested / 1e6:.0f} MB int8 vs ~{full / 1e6:.0f} MB fp32 "
        f"(a ~{full / nested:.0f}x cut) — anyone wanting both tiers, upgrading, "
        "or wanting beyond-fluency coverage wins; only mini-only users lose."
    )


def write_summary(repo: Path, langs: list[str], sizes: dict[str, dict]) -> None:
    totals = {
        key: sum(v[key] for v in sizes.values())
        for key in ("mini_bytes", "fluency_bytes", "separate_bytes", "nested_bytes", "full_bytes")
    }
    summary = {
        "languages": langs,
        "per_language": {
            lang: {
                "truncations": info["truncations"],
                "mini_bytes": info["mini_bytes"],
                "fluency_bytes": info["fluency_bytes"],
                "separate_bytes": info["separate_bytes"],
                "nested_bytes": info["nested_bytes"],
                "full_bytes": info["full_bytes"],
            }
            for lang, info in sizes.items()
        },
        "totals": totals,
        "deltas": {
            "nested_minus_separate_bytes": totals["nested_bytes"] - totals["separate_bytes"],
            "nested_vs_mini_ratio": round(totals["nested_bytes"] / totals["mini_bytes"], 2),
            "nested_vs_full_ratio": round(totals["nested_bytes"] / totals["full_bytes"], 3),
        },
        "recommendation": recommendation(sizes),
        "note": "Experiment only (plan 71): registry.json, manifest.json and release workflow unchanged; parity evidence in eval/reports/nested.parity.json.",
        "generated_at": build_tiers.iso_now(),
    }
    out = repo / "eval" / "reports" / "nested.summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build nested (2D-Matryoshka) single-artifact models (plan 71)")
    parser.add_argument("--lang", nargs="+", help="language codes, e.g. en de")
    parser.add_argument("--all", action="store_true", help="build every language in manifest.json")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    known = build_tiers.manifest_languages(repo)
    if args.all:
        langs = known
    elif args.lang:
        unknown = [l for l in args.lang if l not in known]
        if unknown:
            parser.error(f"unknown languages {unknown}; manifest has {known}")
        langs = args.lang
    else:
        parser.error("specify --lang or --all")

    sizes: dict[str, dict] = {}
    for lang in langs:
        sizes[lang] = build_language(repo, lang)
    write_summary(repo, langs, sizes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
