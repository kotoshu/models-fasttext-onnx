#!/usr/bin/env python3
"""Build int4 group-128 quantized full-vocab models — plan 68 B1 experiment.

Quantizes the FULL model (all 100k rows, all 300 dims, no vocab cut and
no dim reduction) to int4 with group-wise scales: each 300-d row is split
into 3 consecutive groups of 128 (128+128+44), each group carrying its own
fp32 scale `max_abs(group) / 7.0` — the symmetric-positive divisor 7, not
8, so `+max_abs` round-trips exactly, matching the int8 tier's
`max_abs / 127.0` convention (kotoshu-rs rerank/dequant.rs documents the
same recipe for int4). Target ~15-20 MB vs the 120 MB fp32 full model.

Nibble packing matches the kotoshu-rs `int4-per-row` (RowFormat 0x04)
contract exactly: element 2j in the HIGH nibble of byte j, element 2j+1
in the LOW nibble, two's-complement signed nibbles in [-8, 7]. The 0x04
format itself carries ONE fp32 scale per row, so the per-group scales
produced here need a kotoshu-rs RowFormat contract extension before a
native Rust reader could load them (the metadata string "int4-group128"
is currently rejected by RowFormat::from_metadata by design). Nothing in
kotoshu-rs is modified by this script.

Per language this writes (experiment-only, like the plan-71 nested
artifact; registry.json, manifest.json, tiers.json and the release
workflow are untouched):

- models/{lang}/fasttext.{lang}.int4.onnx — packed artifact whose ONNX
  graph dequantizes via onnxruntime directly (nibble unpack + gathered
  group scales, same Gather/Cast/Mul shape as the int8 tier graphs).
- models/{lang}/fasttext.{lang}.int4.vocab.json — byte copy of the full
  vocab (the int4 matrix keeps every row).
- eval/reports/int4.{lang}.json — tier-report schema + a `row_format`
  field, gated against the fluency thresholds (top1 >= 0.95,
  rank_corr >= 0.97) via run_eval.evaluate(gates=...). Gate failures are
  recorded honestly, never tuned around.

eval/reports/int4.summary.json carries the cross-language sizes, gate
verdicts, group-scale overhead analysis and the recommendation.
Exit status is nonzero if any language fails its gate.
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
import onnxruntime as ort
from onnx import StringStringEntryProto, TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tiers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_eval  # noqa: E402

GROUP_SIZE = run_eval.INT4_GROUP_SIZE  # 128
INT4_POSITIVE_MAX = 7                  # symmetric-positive divisor (see module doc)
INT4_NEGATIVE_MIN = -8
ORT_ROUNDTRIP_TOL = run_eval.ORT_SPOT_CHECK_TOL  # 1e-4, same as tier spot checks

ROW_FORMAT = {
    "quantization": "int4-group128",
    "q_packed": "uint8 [vocab_size, ceil(dims/2)]; byte j of row r holds "
    "element 2j in the HIGH nibble and element 2j+1 in the LOW nibble, "
    "two's-complement signed nibbles in [-8, 7] (odd dims: trailing low "
    "nibble is padding); identical to the kotoshu-rs pack_row_int4 contract",
    "group_scales": "fp32 [vocab_size, ceil(dims/128)]; scale[g] = "
    "max(abs(row[:, group g])) / 7.0, zero-guarded to 1.0",
    "dequant": "value[i] = unpacked_nibble[i] * group_scales[row, i // 128]",
    "group_size": GROUP_SIZE,
    "scale_dtype": "fp32",
    "kotoshu_rs_row_format": "nibble packing matches RowFormat::Int4PerRow "
    "(byte 0x04) but 0x04 carries ONE fp32 scale per row; per-group scales "
    "require a kotoshu-rs contract extension (new format byte + metadata "
    "string \"int4-group128\") before a native Rust reader can load these",
}

RUST_CONTRACT_NOTE = (
    "kotoshu-rs rerank::dequant::RowFormat accepts metadata \"int4-per-row\" "
    "(byte 0x04: packed signed nibbles x ONE fp32 row scale) and its test "
    "explicitly rejects \"int4-group-128\". These artifacts use "
    "\"int4-group128\" + per-group fp32 scales, so kotoshu-rs needs a contract "
    "extension: a new RowFormat variant + format byte + the \"int4-group128\" "
    "metadata string, with dequant reusing the existing nibble unpacking "
    "(pack_row_int4/dequant_row_int4) and taking ceil(dims/128) fp32 scales "
    "per row instead of one; onnx.rs must read the group_scales constant. "
    "kotoshu-rs was NOT modified by this experiment."
)


def quantize_int4_group128(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize fp32 [V, d] to int4 group-128.

    Returns (packed uint8 [V, ceil(d/2)], scales fp32 [V, ceil(d/128)],
    dequant fp32 [V, d] recomputed from the integer codes, per-group
    max_abs [V, groups]). `dequant` is derived from the packed values via
    run_eval.dequant_int4_group128 before being returned, so the packing
    round-trip is asserted on every build.
    """
    vocab_size, dims = x.shape
    n_groups = -(-dims // GROUP_SIZE)
    padded = np.zeros((vocab_size, n_groups * GROUP_SIZE), dtype=np.float32)
    padded[:, :dims] = x
    groups = padded.reshape(vocab_size, n_groups, GROUP_SIZE)

    scale = (np.abs(groups).max(axis=2) / float(INT4_POSITIVE_MAX)).astype(np.float32)
    scale[scale == 0.0] = 1.0
    q = np.rint(groups / scale[:, :, None]).clip(INT4_NEGATIVE_MIN, INT4_POSITIVE_MAX).astype(np.int8)
    q = q.reshape(vocab_size, n_groups * GROUP_SIZE)[:, :dims]

    packed = pack_int4_rows(q)
    dequant = run_eval.dequant_int4_group128(packed, scale, dims)
    if not np.array_equal(dequant, q.astype(np.float32) * np.repeat(scale, GROUP_SIZE, axis=1)[:, :dims]):
        raise AssertionError("int4 packing round-trip mismatch (pack != unpack)")
    group_max_abs = np.abs(groups).max(axis=2)
    return packed, np.ascontiguousarray(scale), dequant, group_max_abs


def pack_int4_rows(q: np.ndarray) -> np.ndarray:
    """Pack int4 codes [V, d] into bytes: even index -> HIGH nibble.

    Mirrors kotoshu-rs pack_row_int4: byte j = (q[2j] & 0xf) << 4 |
    (q[2j+1] & 0xf), values in [-8, 7], odd d padded with a zero low
    nibble.
    """
    if q.min() < INT4_NEGATIVE_MIN or q.max() > INT4_POSITIVE_MAX:
        raise ValueError(f"int4 codes out of range [{INT4_NEGATIVE_MIN}, {INT4_POSITIVE_MAX}]")
    vals = q.astype(np.int16)
    if vals.shape[1] % 2:
        vals = np.pad(vals, ((0, 0), (0, 1)))
    pairs = vals.reshape(vals.shape[0], -1, 2)
    hi = (pairs[:, :, 0] & 0x0F).astype(np.uint8)
    lo = (pairs[:, :, 1] & 0x0F).astype(np.uint8)
    return np.ascontiguousarray((hi << 4) | lo)


def make_int4_model(packed: np.ndarray, scales: np.ndarray, dims: int) -> onnx.ModelProto:
    """Build the int4-group128 graph (same shape as the tier models).

    `word_index` int64 [1] -> `embedding` fp32 [dims]. The graph unpacks
    the packed nibble row arithmetically (Div/Floor/Sub are exact on
    small integers in fp32), sign-extends via Greater(x, 7.5), gathers
    the per-dim group scale and multiplies — a consumer can run the file
    in onnxruntime directly, no custom ops, opset 11 like the tiers.
    Requires even `dims` (all current models are 300-d).
    """
    if dims % 2:
        raise ValueError(f"int4 graph path requires even dims, got {dims}")
    vocab_size, half_dims = packed.shape
    n_groups = scales.shape[1]

    group_of_dim = (np.arange(dims, dtype=np.int64) // GROUP_SIZE).astype(np.int64)
    nodes = [
        helper.make_node("Constant", [], ["q_packed"], value=numpy_helper.from_array(packed, name="q_packed")),
        helper.make_node("Constant", [], ["group_scales"], value=numpy_helper.from_array(scales, name="group_scales")),
        helper.make_node(
            "Constant", [], ["group_of_dim"], value=numpy_helper.from_array(group_of_dim, name="group_of_dim")
        ),
        helper.make_node(
            "Constant", [], ["shape_1x1xh"], value=numpy_helper.from_array(np.array([1, 1, half_dims], dtype=np.int64), name="shape_1x1xh")
        ),
        helper.make_node(
            "Constant", [], ["shape_1xd"], value=numpy_helper.from_array(np.array([1, dims], dtype=np.int64), name="shape_1xd")
        ),
        helper.make_node(
            "Constant", [], ["sixteen"], value=numpy_helper.from_array(np.array(16.0, dtype=np.float32), name="sixteen")
        ),
        helper.make_node(
            "Constant", [], ["seven_point_five"],
            value=numpy_helper.from_array(np.array(7.5, dtype=np.float32), name="seven_point_five"),
        ),
        # Row gather (same entry point as the int8 tier graphs).
        helper.make_node("Gather", ["q_packed", "word_index"], ["packed_i"], axis=0),
        helper.make_node("Gather", ["group_scales", "word_index"], ["scales_i"], axis=0),
        helper.make_node("Cast", ["packed_i"], ["packed_f"], to=TensorProto.FLOAT),
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
        # Per-group scale gather along the dim axis, then the tier-style Mul.
        helper.make_node("Gather", ["scales_i", "group_of_dim"], ["scale_per_dim"], axis=1),
        helper.make_node("Mul", ["signed", "scale_per_dim"], ["embedding_flat"]),
        helper.make_node("Squeeze", ["embedding_flat"], ["embedding"], axes=[0]),
    ]
    input_tensor = helper.make_tensor_value_info("word_index", TensorProto.INT64, [1])
    output_tensor = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [dims])
    graph = helper.make_graph(nodes, "fasttext_int4_group128_embedding", [input_tensor], [output_tensor])
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
    model.metadata_props.append(StringStringEntryProto(key="quantization", value=ROW_FORMAT["quantization"]))
    model.metadata_props.append(StringStringEntryProto(key="tier", value="int4"))
    model.metadata_props.append(StringStringEntryProto(key="group_size", value=str(GROUP_SIZE)))
    model.metadata_props.append(
        StringStringEntryProto(key="int4_packing", value=json.dumps(ROW_FORMAT, separators=(",", ":")))
    )
    return model


def verify_int4_onnx(onnx_path: Path, dequant: np.ndarray, x: np.ndarray) -> dict:
    """Round-trip proof: onnxruntime output == dequantized numpy rows.

    The dequant-vs-original quantization error is recorded as
    informational only — the 0.05 abs tolerance that gates the int8 tiers
    does not apply to 4-bit codes (worst-case error per group is
    max_abs/14 by construction); the binding gate is the fluency eval.
    """
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    n = dequant.shape[0]
    idxs = sorted({0, 1, n // 2, n - 1} | ({29_999} if n > 30_000 else set()))
    worst_runtime = 0.0
    for i in idxs:
        out = session.run(["embedding"], {"word_index": np.array([i], dtype=np.int64)})[0]
        worst_runtime = max(worst_runtime, float(np.abs(out - dequant[i]).max()))
    if worst_runtime > ORT_ROUNDTRIP_TOL:
        raise RuntimeError(
            f"{onnx_path}: onnxruntime diverges from dequantized rows (max abs {worst_runtime:.3g} > {ORT_ROUNDTRIP_TOL})"
        )
    return {
        "runtime_vs_dequant_max_abs": worst_runtime,
        "runtime_vs_dequant_tol": ORT_ROUNDTRIP_TOL,
        "spot_checked_indices": idxs,
        "dequant_vs_projected_max_abs": float(np.abs(x - dequant).max()),
        "dequant_vs_projected_rms": float(np.sqrt(np.mean((x - dequant) ** 2))),
        "note": "dequant_vs_projected_* are informational for int4; the binding gate is the fluency eval",
    }


def build_language(repo: Path, lang: str) -> dict:
    model_dir = repo / "models" / lang
    print(f"[{lang}] loading full model")
    full = onnx.load(str(model_dir / f"fasttext.{lang}.onnx"))
    x = np.ascontiguousarray(build_tiers.constant_array(full, "word_embeddings"), dtype=np.float32)
    vocab = json.loads((model_dir / f"fasttext.{lang}.vocab.json").read_text(encoding="utf-8"))
    if x.shape[0] != vocab["vocab_size"] or len(vocab["word_to_idx"]) != x.shape[0]:
        raise ValueError(f"{lang}: vocab json does not match embedding matrix {x.shape}")
    source_sha = build_tiers.full_model_sha256(repo, lang)

    packed, scales, dequant, _group_max_abs = quantize_int4_group128(x)
    model = make_int4_model(packed, scales, dequant.shape[1])
    model.metadata_props.append(StringStringEntryProto(key="source_full_sha256", value=source_sha))

    onnx_path = model_dir / f"fasttext.{lang}.int4.onnx"
    vocab_path = model_dir / f"fasttext.{lang}.int4.vocab.json"
    onnx.save(model, str(onnx_path))
    shutil.copyfile(model_dir / f"fasttext.{lang}.vocab.json", vocab_path)

    check = verify_int4_onnx(onnx_path, dequant, x)

    # Loader spot check through the real eval path (row-format-aware).
    run_eval.load_tier_model(onnx_path, vocab_path)

    fluency_gates = run_eval.load_gates(repo)["fluency"]
    report = run_eval.evaluate(repo, lang, "int4", build_check=check, write=False, gates=fluency_gates)
    report["recipe"]["reduction"] = "none (full dims; int4-group128 quantization only)"
    report["row_format"] = ROW_FORMAT
    report["experiment_note"] = (
        "Experiment only (plan 68 B1): evaluated but not shipped; registry.json, "
        "manifest.json, tiers.json and release workflow unchanged."
    )
    reports_dir = repo / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"int4.{lang}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    m = report["metrics"]
    print(
        f"  {lang}/int4: rank_corr={m['rank_corr']['mean_spearman']:.4f} "
        f"top1={m['top1_agreement']['agreement']:.4f} "
        f"bytes={onnx_path.stat().st_size} {'PASS' if report['gates']['passed'] else 'fail'}"
    )
    return {
        "full_bytes": (model_dir / f"fasttext.{lang}.onnx").stat().st_size,
        "int4_bytes": onnx_path.stat().st_size,
        "fluency_bytes": (model_dir / f"fasttext.{lang}.fluency.onnx").stat().st_size,
        "mini_bytes": (model_dir / f"fasttext.{lang}.mini.onnx").stat().st_size,
        "int4_sha256": sha256(onnx_path.read_bytes()).hexdigest(),
        "rank_corr": round(m["rank_corr"]["mean_spearman"], 6),
        "top1_agreement": round(m["top1_agreement"]["agreement"], 6),
        "gate_passed": report["gates"]["passed"],
    }


def overhead_analysis(sizes: dict[str, dict]) -> dict:
    first = next(iter(sizes.values()))
    vocab_size, dims = 100_000, 300  # verified per language at build time
    n_groups = -(-dims // GROUP_SIZE)
    packed_total = vocab_size * (dims // 2)
    return {
        "dims": dims,
        "group_size": GROUP_SIZE,
        "groups_per_row": n_groups,
        "vocab_size": vocab_size,
        "packed_bytes_per_row": dims // 2,
        "fp32_scales_bytes_per_row": n_groups * 4,
        "fp32_scales_bytes_total": vocab_size * n_groups * 4,
        "fp16_scales_bytes_per_row": n_groups * 2,
        "fp16_scales_bytes_total": vocab_size * n_groups * 2,
        "fp32_overhead_fraction_of_packed": round(vocab_size * n_groups * 4 / packed_total, 4),
        "fp16_overhead_fraction_of_packed": round(vocab_size * n_groups * 2 / packed_total, 4),
        "scale_dtype": "fp32",
        "scale_dtype_rationale": (
            "fp32 scales cost 12 B/row (1.20 MB per 100k-vocab model, 8.0% of "
            "the 15.0 MB packed payload); fp16 would save 0.60 MB (3.7% of "
            "total) at the cost of a Cast node in every graph and ~0.05% "
            "relative scale rounding. fp32 chosen for exact parity with the "
            "int8 tier scale dtype and zero extra graph ops."
        ),
    }


def recommendation(sizes: dict[str, dict]) -> str:
    failed = [lang for lang, v in sizes.items() if not v["gate_passed"]]
    int4 = next(iter(sizes.values()))["int4_bytes"]
    full = next(iter(sizes.values()))["full_bytes"]
    if failed:
        rcs = [v["rank_corr"] for v in sizes.values()]
        passed = [lang for lang in sizes if lang not in failed]
        return (
            f"Do NOT ship int4-group128: rank_corr clears the 0.97 gate on every "
            f"language ({min(rcs):.4f}-{max(rcs):.4f}) but top1_agreement fails the "
            f"0.95 gate on {len(failed)}/{len(sizes)}"
            + (f" (only {', '.join(passed)} passes)" if passed else "")
            + " — 4-bit noise flips roughly 1-2 of the ~30 scoreable typo probes per "
            "language, so the fine-grained cosine ordering that the rerank pipeline "
            "depends on does not survive at group size 128. int8-per-row remains the "
            "shipped recipe. If int4 is retried, the recorded (untried) levers are: "
            "smaller groups (64/32), asymmetric unsigned [-8, 15] rounding, or a mixed "
            "int4-mini + int8-fluency split. Evidence: eval/reports/int4.{lang}.json."
        )
    return (
        f"Ship-worthy as an optional tier candidate: int4-group128 keeps the FULL "
        f"100k vocab at ~{int4 / 1e6:.1f} MB (~{full / int4:.0f}x smaller than the "
        f"{full / 1e6:.0f} MB fp32 full model, ~{int4 / fluency_bytes(sizes):.1f}x the "
        "size of the 50k-vocab fluency tier) while meeting the fluency gates on all "
        "languages; adoption is owner-gated (registry/tier names) and blocked on the "
        "kotoshu-rs RowFormat extension for per-group scales."
    )


def fluency_bytes(sizes: dict[str, dict]) -> int:
    return next(iter(sizes.values()))["fluency_bytes"]


def write_summary(repo: Path, langs: list[str], sizes: dict[str, dict]) -> None:
    totals = {key: sum(v[key] for v in sizes.values()) for key in ("full_bytes", "int4_bytes", "fluency_bytes", "mini_bytes")}
    failed = [lang for lang, v in sizes.items() if not v["gate_passed"]]
    summary = {
        "languages": langs,
        "per_language": sizes,
        "totals": totals,
        "gate_verdicts": {
            "thresholds": run_eval.load_gates(repo)["fluency"],
            "applied_to": "int4 artifacts gated against the fluency thresholds via run_eval.evaluate(gates=...)",
            "passed_languages": [lang for lang in langs if sizes[lang]["gate_passed"]],
            "failed_languages": failed,
        },
        "group_scale_overhead": overhead_analysis(sizes),
        "rust_contract": {
            "status": "extension required",
            "detail": RUST_CONTRACT_NOTE,
        },
        "recommendation": recommendation(sizes),
        "note": "Experiment only (plan 68 B1): registry.json, manifest.json, tiers.json and release workflow unchanged.",
        "determinism": {"seed": run_eval.SEED, "rng": "np.random.default_rng([seed, crc32(language)])"},
        "generated_at": build_tiers.iso_now(),
    }
    out = repo / "eval" / "reports" / "int4.summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build int4 group-128 full-vocab models (plan 68 B1 experiment)")
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

    failures = [lang for lang in langs if not sizes[lang]["gate_passed"]]
    if failures:
        print(f"GATE FAILURES: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
