#!/usr/bin/env python3
"""Build per-row int4 tier candidates (mini4, fluency4) - plan 101 experiment.

Quantizes the SAME vocab cuts as the shipped int8 tiers (mini top-10k,
fluency top-50k, both read from models/{lang}/tiers.json so a ladder
winner of a different size is honored) to int4 with ONE fp16 scale per
row: scale = max_abs(row) / 7.0 rounded to fp16, codes round(value /
scale_as_fp32) clipped to [-8, 7] (nearest-even, the original recipe),
packed two-a-byte with element 2j in the HIGH nibble of byte j - the
kotoshu-rs `rerank::dequant::pack_row_int4` contract (RowFormat byte
0x04 shape: packed signed nibbles x one row scale). fp16 scales are what
plan 101 asks for; the 0x04 contract itself says fp32, so shipping would
require either an fp16 scale-dtype flag or fp32 scales on the Rust
reader side (plan 101 step 4, only on gate pass). Codes are quantized
against the fp16-rounded scale so dequant = code * stored_scale is exact.

Per language and tier this writes (experiment-only until the plan 101
verdict says otherwise; registry.json, manifest.json, tiers.json and the
release workflow are untouched by this script):

- models/{lang}/fasttext.{lang}.{tier4}.onnx - packed artifact whose
  graph dequantizes via onnxruntime directly (nibble unpack + fp16 scale
  Cast + Mul, same Gather/Cast/Mul shape as the tier graphs).
- models/{lang}/fasttext.{lang}.{tier4}.vocab.json - byte copy of the
  matching int8 tier vocab (identical cut, so the eval probe stream and
  scoreable sets are identical to the int8 baseline runs).
- eval/reports/{lang}.{tier4}.json - tier-report schema + row_format +
  int8 baseline comparison, gated against the int8 tier thresholds via
  run_eval.evaluate(gates=...). Gate failures are recorded honestly.

Tier4 names: mini4 (mini cut, mini gates), fluency4 (fluency cut,
fluency gates).

Acceptance (plan 101, stricter than the tier gates): per language and
tier, rank_corr within 0.001 of the int8 tier measured on the same probe
stream AND the top1_agreement gate check holds. The verdict and the
per-language deltas land in eval/reports/int4-per-row.summary.json.
Exit status is nonzero if the acceptance fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import StringStringEntryProto, TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_int4  # noqa: E402  (pack_int4_rows, verify_int4_onnx helpers)
import build_tiers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_eval  # noqa: E402

INT4_POSITIVE_MAX = 7
INT4_NEGATIVE_MIN = -8
RANK_CORR_WITHIN = 0.001  # plan 101 acceptance: within 0.001 of int8

# tier4 name -> (int8 tier whose vocab cut and gates it mirrors)
TIER4_BASE = {"mini4": "mini", "fluency4": "fluency"}


def quantize_int4_per_row(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize fp32 [V, d] to int4 with ONE fp16 scale per row.

    scale = max_abs(row)/7.0 rounded to fp16 (zero-guarded to 1.0);
    codes = rint(x / fp16_scale) clipped to [-8, 7]. Returns (packed
    uint8 [V, ceil(d/2)], scales fp16 [V], dequant fp32 [V, d] recomputed
    from the packed values so the packing round-trip is asserted on
    every build).
    """
    scale32 = (np.abs(x).max(axis=1) / float(INT4_POSITIVE_MAX)).astype(np.float32)
    scale32[scale32 == 0.0] = 1.0
    scales = scale32.astype(np.float16)
    scale_used = scales.astype(np.float32)  # exactly what dequant will use
    q = np.rint(x / scale_used[:, None]).clip(INT4_NEGATIVE_MIN, INT4_POSITIVE_MAX).astype(np.int8)
    packed = build_int4.pack_int4_rows(q)
    dims = x.shape[1]
    values = run_eval.unpack_int4_packed(packed)[:, :dims]
    dequant = values * scale_used[:, None]
    if not np.array_equal(dequant, q.astype(np.float32) * scale_used[:, None]):
        raise AssertionError("int4 packing round-trip mismatch (pack != unpack)")
    return packed, np.ascontiguousarray(scales), np.ascontiguousarray(dequant)


def row_format() -> dict:
    return {
        "quantization": "int4-per-row",
        "q_packed": "uint8 [vocab_size, ceil(dims/2)]; byte j of row r holds "
        "element 2j in the HIGH nibble and element 2j+1 in the LOW nibble, "
        "two's-complement signed nibbles in [-8, 7] (odd dims: trailing low "
        "nibble is padding); identical to the kotoshu-rs pack_row_int4 contract",
        "row_scales": "fp16 [vocab_size]; scale = max(abs(row)) / 7.0 rounded "
        "to fp16, zero-guarded to 1.0",
        "dequant": "value[i] = unpacked_nibble[i] * row_scales[row]",
        "rounding": "nearest-even",
        "scale_dtype": "fp16",
        "kotoshu_rs_row_format": 'nibble packing and one-scale-per-row match '
        "RowFormat::Int4PerRow (byte 0x04, metadata int4-per-row) but that "
        "contract carries ONE fp32 scale per row; fp16 scales need a "
        "scale-dtype flag (or fp32 scales) on the Rust reader side before a "
        "native load - plan 101 step 4, gated on this experiment passing",
    }


def make_int4_per_row_model(
    packed: np.ndarray, scales: np.ndarray, dims: int, tier: str
) -> onnx.ModelProto:
    """Build the int4-per-row graph (same shape as the tier models).

    `word_index` int64 [1] -> `embedding` fp32 [dims]. The graph unpacks
    the packed nibble row arithmetically (Div/Floor/Sub are exact on
    small integers in fp32), sign-extends via Greater(x, 7.5), casts the
    fp16 row scale to fp32 and multiplies - a consumer can run the file
    in onnxruntime directly, no custom ops, opset 11 like the tiers.
    Requires even `dims` (all current models are 300-d).
    """
    if dims % 2:
        raise ValueError(f"int4 graph path requires even dims, got {dims}")
    vocab_size, half_dims = packed.shape

    nodes = [
        helper.make_node("Constant", [], ["q_packed"], value=numpy_helper.from_array(packed, name="q_packed")),
        helper.make_node("Constant", [], ["row_scales"], value=numpy_helper.from_array(scales, name="row_scales")),
        helper.make_node(
            "Constant", [], ["shape_1x1xh"],
            value=numpy_helper.from_array(np.array([1, 1, half_dims], dtype=np.int64), name="shape_1x1xh"),
        ),
        helper.make_node(
            "Constant", [], ["shape_1xd"],
            value=numpy_helper.from_array(np.array([1, dims], dtype=np.int64), name="shape_1xd"),
        ),
        helper.make_node(
            "Constant", [], ["shape_1x1"],
            value=numpy_helper.from_array(np.array([1, 1], dtype=np.int64), name="shape_1x1"),
        ),
        helper.make_node(
            "Constant", [], ["sixteen"], value=numpy_helper.from_array(np.array(16.0, dtype=np.float32), name="sixteen"),
        ),
        helper.make_node(
            "Constant", [], ["seven_point_five"],
            value=numpy_helper.from_array(np.array(7.5, dtype=np.float32), name="seven_point_five"),
        ),
        # Row gather (same entry point as the int8 tier graphs).
        helper.make_node("Gather", ["q_packed", "word_index"], ["packed_i"], axis=0),
        helper.make_node("Gather", ["row_scales", "word_index"], ["scales_i"], axis=0),
        helper.make_node("Cast", ["packed_i"], ["packed_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["scales_i"], ["scales_f"], to=TensorProto.FLOAT),
        # Unpack: high nibble = floor(byte / 16), low nibble = byte - 16*high.
        helper.make_node("Div", ["packed_f", "sixteen"], ["over_sixteen"]),
        helper.make_node("Floor", ["over_sixteen"], ["hi_u"]),
        helper.make_node("Mul", ["hi_u", "sixteen"], ["hi_times_sixteen"]),
        helper.make_node("Sub", ["packed_f", "hi_times_sixteen"], ["lo_u"]),
        # Interleave high/low nibbles back into element order.
        helper.make_node("Reshape", ["hi_u", "shape_1x1xh"], ["hi_r"]),
        helper.make_node("Reshape", ["lo_u", "shape_1x1xh"], ["lo_r"]),
        helper.make_node("Concat", ["hi_r", "lo_r"], ["hl"], axis=1),
        helper.make_node("Transpose", ["hl"], ["lh"], perm=[0, 2, 1]),
        helper.make_node("Reshape", ["lh", "shape_1xd"], ["nib_u"]),
        # Two's-complement sign extension: nibble > 7.5 -> nibble - 16.
        helper.make_node("Greater", ["nib_u", "seven_point_five"], ["neg_b"]),
        helper.make_node("Cast", ["neg_b"], ["neg_m"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["neg_m", "sixteen"], ["neg_sixteen"]),
        helper.make_node("Sub", ["nib_u", "neg_sixteen"], ["signed"]),
        # Per-row fp16 scale (cast to fp32 above), tier-style broadcast Mul.
        helper.make_node("Reshape", ["scales_f", "shape_1x1"], ["s_1x1"]),
        helper.make_node("Mul", ["signed", "s_1x1"], ["embedding_flat"]),
        helper.make_node("Squeeze", ["embedding_flat"], ["embedding"], axes=[0]),
    ]
    input_tensor = helper.make_tensor_value_info("word_index", TensorProto.INT64, [1])
    output_tensor = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [dims])
    graph = helper.make_graph(nodes, f"fasttext_int4_per_row_embedding", [input_tensor], [output_tensor])
    model = helper.make_model(
        graph,
        producer_name="kotoshu-fasttext-converter",
        producer_version="1.0.0",
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=11,
    )
    model.metadata_props.append(StringStringEntryProto(key="vocabulary_size", value=str(vocab_size)))
    model.metadata_props.append(StringStringEntryProto(key="embedding_dimension", value=str(dims)))
    model.metadata_props.append(StringStringEntryProto(key="model_type", value="fasttext_embedding"))
    model.metadata_props.append(StringStringEntryProto(key="quantization", value="int4-per-row"))
    model.metadata_props.append(StringStringEntryProto(key="tier", value=tier))
    model.metadata_props.append(StringStringEntryProto(key="scale_dtype", value="fp16"))
    model.metadata_props.append(
        StringStringEntryProto(key="int4_packing", value=json.dumps(row_format(), separators=(",", ":")))
    )
    return model


def build_language_tier(repo: Path, lang: str, tier4: str) -> dict:
    """Build and gate one {lang}/{tier4} candidate against its int8 twin."""
    base = TIER4_BASE[tier4]
    model_dir = repo / "models" / lang

    tiers = json.loads((model_dir / "tiers.json").read_text(encoding="utf-8"))["tiers"]
    vocab_size = tiers[base]["vocab_size"]  # SAME vocab cut as the int8 tier

    print(f"[{lang}/{tier4}] loading full model, cutting to top-{vocab_size}")
    full = onnx.load(str(model_dir / f"fasttext.{lang}.onnx"))
    x_all = np.ascontiguousarray(build_tiers.constant_array(full, "word_embeddings"), dtype=np.float32)
    vocab = json.loads((model_dir / f"fasttext.{lang}.vocab.json").read_text(encoding="utf-8"))
    if x_all.shape[0] != vocab["vocab_size"] or len(vocab["word_to_idx"]) != x_all.shape[0]:
        raise ValueError(f"{lang}: vocab json does not match embedding matrix {x_all.shape}")
    source_sha = build_tiers.full_model_sha256(repo, lang)

    x = np.ascontiguousarray(x_all[:vocab_size])
    packed, scales, dequant = quantize_int4_per_row(x)
    model = make_int4_per_row_model(packed, scales, dequant.shape[1], tier=tier4)
    model.metadata_props.append(StringStringEntryProto(key="source_full_sha256", value=source_sha))

    onnx_path = model_dir / f"fasttext.{lang}.{tier4}.onnx"
    vocab_path = model_dir / f"fasttext.{lang}.{tier4}.vocab.json"
    onnx.save(model, str(onnx_path))
    shutil.copyfile(model_dir / f"fasttext.{lang}.{base}.vocab.json", vocab_path)

    check = build_int4.verify_int4_onnx(onnx_path, dequant, x)
    run_eval.load_tier_model(onnx_path, vocab_path)  # loader spot check through the real eval path

    gates = run_eval.load_gates(repo)[base]
    report = run_eval.evaluate(repo, lang, tier4, build_check=check, write=False, gates=gates)
    report["recipe"]["reduction"] = "none (full 300 dims; int4-per-row quantization only)"
    report["row_format"] = row_format()
    report["experiment_note"] = (
        "Experiment only (plan 101): evaluated against the shipped int8 tier "
        "thresholds and its measured numbers; registry.json, manifest.json, "
        "tiers.json and release workflow unchanged."
    )

    # int8 baseline measured on the identical probe stream (same seed, same
    # vocab size, same probe order) - the deltas below are apples-to-apples.
    base_report = json.loads((repo / "eval" / "reports" / f"{lang}.{base}.json").read_text(encoding="utf-8"))
    int8_rc = base_report["metrics"]["rank_corr"]["mean_spearman"]
    int8_top1 = base_report["metrics"]["top1_agreement"]["agreement"]
    rc = report["metrics"]["rank_corr"]["mean_spearman"]
    top1 = report["metrics"]["top1_agreement"]["agreement"]
    within = rc >= int8_rc - RANK_CORR_WITHIN
    top1_holds = report["gates"]["checks"]["top1_agreement"]
    accepted = bool(within and top1_holds and report["gates"]["passed"])
    report["int8_baseline"] = {
        "tier": base,
        "report": f"eval/reports/{lang}.{base}.json",
        "rank_corr": int8_rc,
        "top1_agreement": int8_top1,
        "top1_n_probes_scored": base_report["metrics"]["top1_agreement"]["n_probes_scored"],
    }
    report["plan101_acceptance"] = {
        "rank_corr_within_0.001_of_int8": bool(within),
        "rank_corr_delta": rc - int8_rc,
        "top1_gate_holds": bool(top1_holds),
        "top1_delta": top1 - int8_top1,
        "accepted": accepted,
    }

    reports_dir = repo / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{lang}.{tier4}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    base_bytes = (model_dir / f"fasttext.{lang}.{base}.onnx").stat().st_size
    print(
        f"  {lang}/{tier4}: rank_corr={rc:.6f} (int8 {int8_rc:.6f}, delta {rc - int8_rc:+.6f}) "
        f"top1={top1:.4f} (int8 {int8_top1:.4f}) bytes={onnx_path.stat().st_size} vs int8 {base_bytes} "
        f"{'ACCEPT' if accepted else 'reject'}"
    )
    return {
        "tier": tier4,
        "vocab_size": vocab_size,
        "int4_bytes": onnx_path.stat().st_size,
        "int8_bytes": base_bytes,
        "vocab_bytes": vocab_path.stat().st_size,
        "sha256": sha256(onnx_path.read_bytes()).hexdigest(),
        "rank_corr": round(rc, 6),
        "int8_rank_corr": round(int8_rc, 6),
        "rank_corr_delta": round(rc - int8_rc, 6),
        "top1_agreement": round(top1, 6),
        "int8_top1_agreement": round(int8_top1, 6),
        "top1_delta": round(top1 - int8_top1, 6),
        "top1_n_probes_scored": report["metrics"]["top1_agreement"]["n_probes_scored"],
        "gate_passed": report["gates"]["passed"],
        "within_0.001_of_int8": bool(within),
        "accepted": accepted,
    }


def verdict(per_language: dict[str, dict[str, dict]]) -> dict:
    rejected = [
        f"{lang}/{tier4}"
        for lang, tiers in per_language.items()
        for tier4, v in tiers.items()
        if not v["accepted"]
    ]
    if rejected:
        worst = min(v["rank_corr_delta"] for tiers in per_language.values() for v in tiers.values())
        return {
            "accepted": False,
            "text": (
                f"REJECT int4-per-row as a shipping tier: {len(rejected)} of "
                f"{sum(len(t) for t in per_language.values())} language/tier candidates miss the plan 101 "
                "acceptance (rank_corr within 0.001 of the int8 tier on the identical probe "
                f"stream; worst delta {worst:+.6f}). int8-per-row remains the shipped recipe. "
                "Measured, not assumed - same discipline as the SVD rejection."
            ),
        }
    return {
        "accepted": True,
        "text": (
            "All language/tier candidates land within 0.001 rank_corr of int8 and hold "
            "top1 agreement - prepare registry v1.4.0 additive entries (plan 101 step 3)."
        ),
    }


def write_summary(repo: Path, langs: list[str], per_language: dict[str, dict[str, dict]]) -> dict:
    v = verdict(per_language)
    summary = {
        "purpose": "Plan 101: per-row int4 (packed nibbles) + fp16 row scale over the same vocab cuts as fluency/mini, gated against the unchanged eval and the int8 tier measurements.",
        "row_format": row_format(),
        "acceptance_rule": {
            "rank_corr_within_of_int8": RANK_CORR_WITHIN,
            "top1": "gate check must hold (mini >= 0.85, fluency >= 0.95, thresholds unchanged)",
            "comparability": "same seeded probe stream and identical vocab as the int8 tier runs, so deltas are apples-to-apples",
        },
        "languages": langs,
        "per_language": per_language,
        "size_audit_mb": {
            lang: {
                tier4: {
                    "int4_onnx": round(v["int4_bytes"] / 1e6, 2),
                    "int8_onnx": round(v["int8_bytes"] / 1e6, 2),
                    "vocab_json": round(v["vocab_bytes"] / 1e6, 2),
                }
                for tier4, v in tiers.items()
            }
            for lang, tiers in per_language.items()
        },
        "verdict": v,
        "note": "Experiment only (plan 101): registry.json, manifest.json, tiers.json and the release workflow unchanged.",
        "determinism": {"seed": run_eval.SEED, "rng": "np.random.default_rng([seed, crc32(language)])"},
        "generated_at": build_tiers.iso_now(),
    }
    out = repo / "eval" / "reports" / "int4-per-row.summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-row int4 tier candidates (plan 101 experiment)")
    parser.add_argument("--lang", nargs="+", required=True, help="language codes, e.g. en de es fr ru pt it pl")
    parser.add_argument("--tiers", nargs="+", default=list(TIER4_BASE), choices=sorted(TIER4_BASE),
                        help="tier4 variants to build (default: both)")
    parser.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    known = build_tiers.manifest_languages(repo)
    unknown = [l for l in args.lang if l not in known]
    if unknown:
        parser.error(f"unknown languages {unknown}; manifest has {known}")

    per_language: dict[str, dict[str, dict]] = {}
    for lang in args.lang:
        per_language[lang] = {tier4: build_language_tier(repo, lang, tier4) for tier4 in args.tiers}
    summary = write_summary(repo, args.lang, per_language)

    if not summary["verdict"]["accepted"]:
        print("PLAN 101 VERDICT: reject (measured numbers in eval/reports/int4-per-row.summary.json)", file=sys.stderr)
        return 1
    print("plan 101 acceptance met on every language/tier")
    return 0


if __name__ == "__main__":
    sys.exit(main())
